import typing
import collections.abc as cabc

from functools import partial
from typing import ClassVar, Callable, Any, Optional, TypedDict, TypeVar, Union

import jax
import jax.numpy as jnp
import numpy as np

from jax import Array

from .csc import csc_matvec


class BasisLike(typing.Protocol, cabc.Hashable):
    width: np.int32
    depth: np.int32
    degree_begin: np.ndarray[np.int64.dtype]

    def size(self) -> int: ...


class EmptyStaticArgs(TypedDict): ...


OperationT = TypeVar("OperationT")


def _batched_fallback_wrapper(single_tensor_fn):
    """
    Generate a batched tensor function from a function that operates on a single tensor.

    Batches can be multidimensional so this reshapes data into 1D so a single vmap can be used
    before restoring to the original batch shape.

    For example, a tensor with width 2, depth 2 will have data size 7. Then given a batch shape
    of (5,3) each of the original arg shapes will be (5,3,7). Those are then reshaped into
    flat_args, with each element being shape (15,7) so they can be vmapped into flat_result.
    After computing flat_result, it is then shaped back into (5,3,7).

    :param single_tensor_fn: fallback function that takes multiple array args with same dims.
    """

    def wrapped(*args, **kwargs):
        # Reshape args to flat shape ready for vmap
        batch_shape = args[0].shape[:-1]
        flat_args = [arg.reshape((-1, arg.shape[-1])) for arg in args]

        # Compute per tensor then restore original batch shape
        vmapped_fn = jax.vmap(partial(single_tensor_fn, **kwargs))
        flat_result_tuple = vmapped_fn(*flat_args)

        # Unpack the resulting tuple args into original batch shapes
        result = tuple(
            flat_result_arg.reshape((*batch_shape, flat_result_arg.shape[-1]))
            for flat_result_arg in flat_result_tuple
        )
        return result

    return wrapped


class Operation:
    """
    Represents a base class for defining JAX-based operations with support for
    platform-specific implementations, fallback mechanisms, static argument
    construction, and FFI calls.

    This class provides a structured framework for efficiently handling rough-path
    extensions to the set of operations provided by JAX. It includes mechanisms to
    register new operations, manage platform-specific implementations, define
    fallback procedures, and prepare attributes required for FFI calls. The
    functionality is built to support extensibility and should primarily be
    subclassed.

    :cvar no_acceleration: Indicates whether to disable the use of accelerated
        routines and fall back to pure JAX implementations.
    :type no_acceleration: ClassVar[bool]

    :cvar data_layout: Defines the supported data layout for algebra objects.
        Currently, only dense representation is supported.
    :type data_layout: ClassVar[str]

    :cvar fn_name: Stores the unique name of the specific operation.
    :type fn_name: ClassVar[str]

    :cvar supported_platforms: A set of platform names indicating platforms
        with accelerator support for at least one data type.
    :type supported_platforms: ClassVar[set[str]]

    :cvar implementations: A dictionary mapping platform and data type tuples
        to operation function names.
    :type implementations: ClassVar[dict[tuple[str, str], str]]

    :cvar default_ffi_call_args: Default arguments for FFI calls when generating
        platform-specific lowerings of the operation.
    :type default_ffi_call_args: ClassVar[dict[str, Any]]

    :cvar StaticArgs: TypedDict subclass that describes static arguments required
        for FFI implementations and fallback operations.
    :type StaticArgs: ClassVar[type[TypedDict]]

    :ivar basis: The primary basis object associated with the operation.
    :type basis: BasisLike

    :ivar data_dtype: The data type for all input and output data for the operation.
    :type data_dtype: jnp.dtype

    :ivar batch_dims: Configuration of batching dimensions for the current
        operation instance.
    :type batch_dims: tuple[int, ...]

    :ivar static_args: Dictionary of static arguments passed to FFI calls or
        fallback implementations.
    :type static_args: type[TypedDict]

    :ivar result_shape_dtypes: Shape and type information for the output arrays,
        used in FFI call generation.
    :type result_shape_dtypes: tuple[jax.ShapeDtypeStruct, ...]

    :ivar ffi_call_args: Additional arguments provided for FFI calls.
    :type ffi_call_args: dict[str, Any]
    """

    # Flag to disable the use of accelerated routines in any operation entirely
    # and fall back to using the pure JAX implementations
    #
    # Users should not use this option, it is mostly intended for testing
    # and benchmarking purposes
    no_acceleration: ClassVar[bool] = False

    # The set of all operations defined. This is automatically populated
    # when deriving from this class.
    #
    # Users should not interact with this directly
    __all_operations: ClassVar[dict[tuple[str, str], type[OperationT]]] = {}

    # The supported layout for data for algebra objects. At the moment all
    # operations only support densely represented objects. In the future,
    # we might need
    data_layout: ClassVar[str] = "dense"

    # The name of the operation. Each implementation should follow the naming
    # convention described above
    fn_name: ClassVar[str]

    # Set of all platforms that have accelerator support for at least one
    # data type
    supported_platforms: ClassVar[set[str]]

    # The set of implementations operations for each operation. This is a mapping
    # from platform, dtype tuple to function name. The function name should
    # follow the format described above.
    implementations: ClassVar[dict[tuple[str, str], str]]

    # default arguments to be passed to jax.ffi.ffi_call when generating the
    # platform-specific lowering of the operation.
    default_ffi_call_args: ClassVar[dict[str, Any]] = {
        "vmap_method": "broadcast_all",
    }

    # A description of the static arguments that are passed to both the FFI
    # implementors of the operation and the fallback (if it has one). This
    # should not contain the static arguments provided via the basis. These
    # are handled separately as it might be specific to each operation.
    #
    # This should be a TypedDict instance, containing a complete description
    # of all the required and optional arguments. This will be passed to the
    # FFI calls by ** unpacking. Using a TypedDict gives some level of
    # argument checking
    StaticArgs: ClassVar[type[TypedDict]]

    ## The following instance attributes are used by the class upon call to
    ## select from available implementations and populate static arguments.

    # The primary basis associated with the operation
    basis: BasisLike
    # The bases for each argument
    bases: tuple[BasisLike, ...]
    # data type for all data inputs and outputs
    data_dtype: jnp.dtype
    # the configuration of batching
    batch_dims: tuple[int, ...]
    # dictionary of static arguments
    static_args: type[TypedDict]

    # For FFI calls, the shape of the output array(s)
    result_shape_dtypes: tuple[jax.ShapeDtypeStruct, ...]
    # Additional arguments to be passed to the ffi call
    ffi_call_args: dict[str, Any]

    @classmethod
    def register(
        cls,
        platform: str,
        name: str,
        fn_ptr: Any,
        supported_dtypes: set[jnp.dtype],
        ffi_register_kwargs: dict[str, Any],
    ):
        cls.supported_platforms.add(platform)

        for dtype in supported_dtypes:
            key = (platform, str(dtype))
            if key in cls.implementations:
                raise ValueError(
                    f"Implementation {key} already registered for operation {cls.__name__}"
                )
            cls.implementations[key] = name

        jax.ffi.register_ffi_target(
            name, fn_ptr, platform=platform, **ffi_register_kwargs
        )

    @classmethod
    def register_all(
        cls,
        platform: str,
        ops: dict[str, Any],
        supported_dtypes: set[str],
        ffi_register_kwargs: dict[str, Any],
    ):
        dtypes = {jnp.dtype(tp) for tp in supported_dtypes}
        for op_cls in Operation.__all_operations.values():
            name = f"{platform}_{cls.data_layout}_{op_cls.fn_name}"
            if (fn_ptr := ops.get(name, None)) is not None:
                op_cls.register(platform, name, fn_ptr, dtypes, ffi_register_kwargs)

    @classmethod
    def get_operation(
        cls, fn_name: str, layout: str = "dense"
    ) -> Optional[type[OperationT]]:
        """
        Retrieves a registered operation class based on the function name and layout.

        This method attempts to fetch an operation class that corresponds to the given
        combination of function name and layout. If a matching operation is found in the
        registry, it will be returned. Otherwise, it returns None.

        :param fn_name: The name of the function associated with the operation.
        :type fn_name: str
        :param layout: The layout type for the operation, with a default value of "dense".
        :type layout: str
        :return: The operation class matching the specified function name and layout,
            or None if not found.
        :rtype: Optional[type[Operation]]
        """
        key = (fn_name, layout)
        return Operation.__all_operations.get(key, None)

    @classmethod
    def make_result_dtypes(cls, basis, dtype, batch_dims):
        """
        Creates result dtypes for use in computations based on the provided basis, data type, and batch dimensions.

        This method generates a shape and dtype structure that matches the expected result of a computation.
        The shape is determined by combining the batch dimensions with the size of the basis. The dtype specifies
        the data type of the elements within this structure.

        :param basis: Basis object containing size information.
        :type basis: Any
        :param dtype: The data type of the computation results.
        :type dtype: Any
        :param batch_dims: A tuple representing the batch dimensions to include in the shape.
        :type batch_dims: tuple
        :return: A tuple containing a shape and dtype structure for the computation.
        :rtype: tuple
        """
        return (jax.ShapeDtypeStruct(batch_dims + (basis.size(),), dtype),)

    @classmethod
    def get_result_basis(
        cls, bases: tuple[BasisLike, ...], preferred_basis
    ) -> BasisLike:
        """
        Determines the appropriate basis from a list of bases, considering an optional
        preferred basis. The method ensures that all bases in the list have matching
        widths, and it selects the basis with the greatest depth if no preferred basis
        is provided. If a preferred basis is supplied and valid, it returns that basis.

        :param bases: A tuple of BasisLike objects. These represent the candidate bases
                      to be considered. The method ensures all bases have the same width
                      and selects the deepest valid basis.
        :param preferred_basis: An optional BasisLike object. If supplied and valid, this
                                basis will be returned instead of evaluating the others.
        :return: The selected BasisLike object, either the preferred basis (if provided
                 and valid) or the valid deepest basis from the `bases` tuple.
        :raises ValueError: If the `bases` tuple is empty or if any basis in the tuple
                            does not match the width of the first basis. Also raised if
                            the `preferred_basis` width does not match the base width.
        """
        if not bases and preferred_basis is None:
            raise ValueError("basis list should be non-empty")

        choice, *other = bases

        if preferred_basis is not None and choice.width != preferred_basis.width:
            raise ValueError(
                f"mismatched width on basis 0, expected {preferred_basis.width} but got {choice.width}"
            )

        for i, basis in enumerate(other, start=1):
            if basis.width != choice.width:
                raise ValueError(
                    f"mismatched width on basis {i}, expected {choice.width} but got {basis.width}"
                )

            if basis.depth >= choice.depth:
                choice = basis

        if preferred_basis is not None:
            return preferred_basis

        return choice

    @classmethod
    def __init_subclass__(cls, **kwargs):
        if not hasattr(cls, "supported_platforms"):
            cls.supported_platforms = set()

        if not hasattr(cls, "implementations"):
            cls.implementations = {}

        if not hasattr(cls, "StaticArgs"):
            cls.StaticArgs = EmptyStaticArgs

        Operation.__all_operations[cls.fn_name, cls.data_layout] = cls

    def __init__(
        self,
        bases,
        dtype,
        batch_dims,
        ffi_call_args: Optional[dict[str, Any]] = None,
        specific_basis: Optional[BasisLike] = None,
        **kwargs,
    ):
        self.basis = basis = self.get_result_basis(bases, specific_basis)
        self.bases = bases
        self.data_dtype = dtype
        self.batch_dims = batch_dims

        self.result_shape_dtypes = self.make_result_dtypes(basis, dtype, batch_dims)
        self.ffi_call_args = self.default_ffi_call_args | (ffi_call_args or {})
        self.static_args = self.make_static_args(kwargs)

    def make_static_args(self, kwargs) -> type[TypedDict]:
        """
        Construct static args from class kwargs.

        Defaults to struct of args. Can be overridden by operations where static
        data needs pre-processing before work is handed over to FFI or fallback.
        """
        return self.StaticArgs(**kwargs)

    def get_fallback(self) -> Callable[..., Any]:
        """
        Provides a mechanism to retrieve a fallback callable with pre-configured
        static parameters. If a fallback callable is not defined, raises an error
        indicating the absence of a fallback operation.

        :raises AttributeError: Raised when no fallback operation has been defined.
        :return: A callable object configured with the fallback operation and
            associated static parameters.
        :rtype: Callable[..., Any]
        """
        if (fb := getattr(self, "fallback", None)) is not None:
            fallback = _batched_fallback_wrapper(fb)
            return partial(fallback, **self.make_ffi_static_args())

        raise AttributeError("{type(self)} does not name a fallback operation")

    def make_ffi_static_args(self) -> dict:
        """
        Creates a dictionary of arguments containing static attributes for FFI (Foreign Function
        Interface) with specific keys and values derived from basis properties and additional static
        arguments.

        :return: A dictionary containing static arguments with keys 'width', 'depth', and
            'degree_begin', derived from the basis attributes, along with additional static
            arguments merged from `self.static_args`.
        :rtype: dict
        """
        return {
            "width": np.int32(self.basis.width),
            "depth": np.int32(self.basis.depth),
            "degree_begin": self.basis.degree_begin,
            **self.static_args,
        }

    def make_ffi_call(self, name) -> Callable[..., Any]:
        """
        Constructs and returns a callable function that performs an FFI (Foreign Function Interface) call
        using the specified name and the associated attributes of the current instance.

        The generated callable uses the `jax.ffi.ffi_call` to execute the operation, passing the configured
        parameters of this class instance, including result shape and data types, custom FFI arguments,
        and static parameters. The callable accepts additional positional and keyword arguments at
        runtime.

        :param name: The name of the target FFI function to call.
        :type name: str
        :return: A callable function that performs the FFI operation with the provided arguments.
        :rtype: Callable[..., Any]
        """
        ffi_static_args = self.make_ffi_static_args()

        def func(*args):
            return jax.ffi.ffi_call(
                name, self.result_shape_dtypes, **self.ffi_call_args
            )(*args, **ffi_static_args)

        return func

    def get_implementations(self) -> dict[str, Callable[..., Any]]:
        """
        Retrieve platform-specific implementations based on the data type.

        This method constructs a dictionary of available implementations for each
        supported platform, filtered by the data type associated with the current
        object instance.

        :param self: The instance of the class containing the method.
        :return: A dictionary where keys are platform names (str) and values are
            callable implementations (Callable[..., Any]) corresponding to the
            platform and data type.
        :rtype: dict[str, Callable[..., Any]]
        """
        dtype_str = str(self.data_dtype)

        def get_impl(plat):
            return self.implementations.get((plat, dtype_str), None)

        return {
            platform: self.make_ffi_call(impl_name)
            for platform in self.supported_platforms
            if (impl_name := get_impl(platform)) is not None
        }

    def convert_args_dtypes(self, *data_args):
        """
        Converts the data types of the provided arguments to match the instance's data
        dtype attribute. This operation ensures the consistency of data types within
        the instance's context.

        :param data_args: Positional arguments whose data types need to be converted.
        :type data_args: tuple
        :return: A tuple containing arguments with their data types converted to
            the instance's `data_dtype` data type.
        :rtype: tuple
        """
        return tuple(arg.astype(self.data_dtype) for arg in data_args)

    def __call__(self, *data_args) -> cabc.Sequence[Array]:
        # Most operations will require homogeneous data arguments
        converted_args = self.convert_args_dtypes(*data_args)

        # The fallback implementation, preloaded with static args
        fallback = self.get_fallback()

        if self.no_acceleration:
            return fallback(*converted_args)

        impls = self.get_implementations()
        return jax.lax.platform_dependent(*converted_args, **impls, default=fallback)


class DenseOperation:
    data_layout = "dense"


## Basic operations
def _dense_ft_mul_level_accumulator(
    lhs_data: Array,
    rhs_data: Array,
    out_degree: np.int32,
    degree_begin: np.ndarray[np.int64.dtype],
    lhs_max_degree: np.int32,
    rhs_max_degree: np.int32,
    lhs_min_degree: np.int32 = 0,
    rhs_min_degree: np.int32 = 0,
):
    """
    Compute the accumulated multiplication for a level of the free tensor at out_degree

    For a free tensor of say width 2 and depth 2, degree_begin will be [0,1,3,7]. The first
    level would be 0-1, second 1-3, third 3-7. Each of these is computed independently by
    this method and concatenated together into a new free tensor. For example, given
    out_degree = 2, this method will iterates lhs_deg over 0,1,2 using the corresponding
    rhs_deg that contributes to degree 2, i.e rhs_deg over 2,1,0. The flattened outer
    product of each level accumulates into the slice level, for out degree 2, [3,7].
    """
    out_b, out_e = degree_begin[out_degree], degree_begin[out_degree + 1]
    out_size = out_e - out_b

    acc = jnp.zeros((out_size,), dtype=lhs_data.dtype)

    lhs_deg_b = max(lhs_min_degree, out_degree - rhs_max_degree)
    lhs_deg_e = min(lhs_max_degree, out_degree - rhs_min_degree) + 1

    for lhs_deg in range(lhs_deg_b, lhs_deg_e):
        # Slice for each lhs level
        lhs_b, lhs_e = degree_begin[lhs_deg], degree_begin[lhs_deg + 1]
        lhs_level = lhs_data[lhs_b:lhs_e]

        # Corresponding rhs slice to fit into out_degree
        rhs_deg = out_degree - lhs_deg
        rhs_b, rhs_e = degree_begin[rhs_deg], degree_begin[rhs_deg + 1]
        rhs_level = rhs_data[rhs_b:rhs_e]

        # Accumulate flattened outer product of two 1D arrays
        frag = jnp.kron(lhs_level, rhs_level)
        acc = acc + frag

    return acc


def _fallback_dense_ft_mul(
    lhs_data: Array,
    rhs_data: Array,
    degree_begin: np.ndarray[np.int64.dtype],
    lhs_max_degree: np.int32,
    rhs_max_degree: np.int32,
    out_max_degree: np.int32,
    lhs_min_degree: np.int32 = 0,
    rhs_min_degree: np.int32 = 0,
):
    """
    Multiply two dense free tensors with given data and degree

    Convenience wrapper that concatenates individual outer products of data
    at each level into one tensor.
    """
    level_gen = partial(
        _dense_ft_mul_level_accumulator,
        lhs_data=lhs_data,
        rhs_data=rhs_data,
        degree_begin=degree_begin,
        lhs_min_degree=lhs_min_degree,
        lhs_max_degree=lhs_max_degree,
        rhs_min_degree=rhs_min_degree,
        rhs_max_degree=rhs_max_degree,
    )

    mul = jnp.concatenate(
        [level_gen(out_degree=i) for i in range(0, out_max_degree + 1)], axis=-1
    )

    return mul


def _fallback_dense_ft_exp(
    arg_data: Array, degree_begin: np.ndarray[np.int64.dtype], arg_max_deg: np.int32
):
    # Use Horner's method to compute exp(x) = Σ((x^n) / n!)
    out = jnp.zeros_like(arg_data).at[0].add(1)

    for deg in range(arg_max_deg, 0, -1):
        max_level = arg_max_deg - deg + 1

        mul = _fallback_dense_ft_mul(
            lhs_data=out,
            rhs_data=arg_data,
            degree_begin=degree_begin,
            out_max_degree=arg_max_deg,
            lhs_min_degree=0,
            lhs_max_degree=max_level,
            rhs_min_degree=1,
            rhs_max_degree=max_level,
        )

        out = mul / deg
        out = out.at[0].add(1)

    return out


def _fallback_dense_antipode(
    arg_data: Array,
    width: np.int32,
    depth: np.int32,
    degree_begin: np.ndarray[np.int64.dtype],
    no_sign: bool = False,
):
    def transpose_level(i):
        sign = 1 if (no_sign or i % 2 == 0) else -1
        level_data = arg_data[degree_begin[i] : degree_begin[i + 1]].reshape(
            (width,) * i
        )
        return sign * jnp.transpose(level_data).reshape(-1)

    antipode = jnp.concatenate([transpose_level(i) for i in range(depth + 1)], axis=-1)

    return antipode


def _fallback_ft_adj_lmul(
    op_data: Array,
    arg_data: Array,
    depth: np.int32,
    degree_begin: np.ndarray[np.int64.dtype],
    op_max_deg: np.int32,
    arg_max_deg: np.int32,
    op_min_deg: np.int32 = 0,
    arg_min_deg: np.int32 = 0,
):
    out_min_deg = 0
    out_max_deg = depth + 1

    out_size = degree_begin[out_max_deg] - degree_begin[out_min_deg]
    out = jnp.zeros((out_size,), dtype=op_data.dtype)

    arg_max_degree = min(arg_max_deg - op_min_deg, out_max_deg)
    arg_min_degree = max(arg_min_deg - op_max_deg, out_min_deg)

    for arg_degree in range(arg_max_degree, arg_min_degree - 1, -1):
        out_min_degree = max(arg_degree - op_max_deg, out_min_deg)
        out_max_degree = min(arg_degree - op_min_deg, out_max_deg)

        arg_frag_lhs = degree_begin[arg_degree]
        arg_frag_rhs = degree_begin[arg_degree + 1]
        arg_frag = arg_data[arg_frag_lhs:arg_frag_rhs]

        for out_degree in range(out_max_degree, out_min_degree - 1, -1):
            op_degree = arg_degree - out_degree

            op_frag_lhs = degree_begin[op_degree]
            op_frag_rhs = degree_begin[op_degree + 1]
            op_frag_size = op_frag_rhs - op_frag_lhs
            op_frag = op_data[op_frag_lhs:op_frag_rhs]

            out_frag_lhs = degree_begin[out_degree]
            out_frag_rhs = degree_begin[out_degree + 1]
            out_frag_size = out_frag_rhs - out_frag_lhs

            for op_idx in range(op_frag_size):
                op_offset = op_idx * out_frag_size
                op_val = op_frag[op_idx]
                for i in range(out_frag_size):
                    out = out.at[out_frag_lhs + i].add(op_val * arg_frag[i + op_offset])

    return out


class DenseFTFma(Operation, DenseOperation):
    fn_name = "ft_fma"

    class StaticArgs(TypedDict):
        a_max_deg: np.int32
        b_max_deg: np.int32
        c_max_deg: np.int32
        b_min_deg: np.int32  # not required
        c_min_deg: np.int32  # not required

    @staticmethod
    def fallback(
        a_data: Array,
        b_data: Array,
        c_data: Array,
        width: np.int32,
        depth: np.int32,
        degree_begin: np.ndarray[np.int64.dtype],
        a_max_deg: np.int32,
        b_max_deg: np.int32,
        c_max_deg: np.int32,
        b_min_deg: np.int32 = 0,
        c_min_deg: np.int32 = 0,
    ) -> tuple[Array, ...]:
        # result = b * c + a
        mul = _fallback_dense_ft_mul(
            b_data,
            c_data,
            degree_begin,
            lhs_max_degree=b_max_deg,
            rhs_max_degree=c_max_deg,
            out_max_degree=a_max_deg,
            lhs_min_degree=b_min_deg,
            rhs_min_degree=c_min_deg,
        )
        return (a_data + mul,)


class DenseFTMul(Operation, DenseOperation):
    fn_name = "ft_mul"

    class StaticArgs(TypedDict):
        out_max_deg: np.int32
        lhs_max_deg: np.int32
        rhs_max_deg: np.int32

    @staticmethod
    def fallback(
        lhs_data: Array,
        rhs_data: Array,
        width: np.int32,
        depth: np.int32,
        degree_begin: np.ndarray[np.int64.dtype],
        out_max_deg: np.int32,
        lhs_max_deg: np.int32,
        rhs_max_deg: np.int32,
        lhs_min_deg: np.int32 = 0,
        rhs_min_deg: np.int32 = 0,
    ) -> tuple[Array]:
        mul = _fallback_dense_ft_mul(
            lhs_data,
            rhs_data,
            degree_begin,
            lhs_max_degree=lhs_max_deg,
            rhs_max_degree=rhs_max_deg,
            out_max_degree=out_max_deg,
            lhs_min_degree=lhs_min_deg,
            rhs_min_degree=rhs_min_deg,
        )
        return (mul,)


class DenseAntipode(Operation, DenseOperation):
    fn_name = "ft_antipode"

    class StaticArgs(TypedDict):
        no_sign: bool

    @staticmethod
    def fallback(
        arg_data: Array,
        width: np.int32,
        depth: np.int32,
        degree_begin: np.ndarray[np.int64.dtype],
        arg_max_deg: np.int32,
        no_sign: bool = False,
    ) -> tuple[Array]:
        antipode = _fallback_dense_antipode(
            arg_data, width, depth, degree_begin, no_sign
        )

        return (antipode,)


class DenseSTFma(Operation, DenseOperation):
    fn_name = "st_fma"

    class StaticArgs(TypedDict):
        a_max_deg: np.int32
        b_max_deg: np.int32
        c_max_deg: np.int32
        b_min_deg: np.int32  # not required
        c_min_deg: np.int32  # not required

    @staticmethod
    def fallback(
        a_data: Array,
        b_data: Array,
        c_data: Array,
        width: np.int32,
        depth: np.int32,
        degree_begin: np.ndarray[np.int64.dtype],
        a_max_deg: np.int32,
        b_max_deg: np.int32,
        c_max_deg: np.int32,
        b_min_deg: np.int32 = 0,
        c_min_deg: np.int32 = 0,
    ) -> tuple[Array]:
        raise NotImplementedError(
            "st_fma is not implemented for native JAX, use CPU backend"
        )


class DenseSTMul(Operation, DenseOperation):
    fn_name = "st_mul"

    class StaticArgs(TypedDict):
        lhs_max_deg: np.int32
        rhs_max_deg: np.int32
        lhs_min_deg: np.int32  # not required
        rhs_min_deg: np.int32  # not required

    @staticmethod
    def fallback(
        b_data: Array,
        c_data: Array,
        width: np.int32,
        depth: np.int32,
        degree_begin: np.ndarray[np.int64.dtype],
        out_max_deg: np.int32,
        lhs_max_deg: np.int32,
        rhs_max_deg: np.int32,
        lhs_min_deg: np.int32 = 0,
        rhs_min_deg: np.int32 = 0,
    ) -> tuple[Array]:
        raise NotImplementedError(
            "st_mul is not implemented for native JAX, use CPU backend"
        )


class DenseFTAdjLeftMul(Operation, DenseOperation):
    fn_name = "ft_adj_lmul"

    class StaticArgs(TypedDict):
        op_max_deg: np.int32
        arg_max_deg: np.int32
        op_min_deg: np.int32  # not required
        arg_min_deg: np.int32  # not required

    @staticmethod
    def fallback(
        op_data: Array,
        arg_data: Array,
        width: np.int32,
        depth: np.int32,
        degree_begin: np.ndarray[np.int64.dtype],
        op_max_deg: np.int32,
        arg_max_deg: np.int32,
        op_min_deg: np.int32 = 0,
        arg_min_deg: np.int32 = 0,
    ) -> tuple[Array]:
        lmul = _fallback_ft_adj_lmul(
            op_data,
            arg_data,
            depth,
            degree_begin,
            op_max_deg,
            arg_max_deg,
            op_min_deg,
            arg_min_deg,
        )
        return (lmul,)


class DenseFTAdjRightMul(Operation, DenseOperation):
    fn_name = "ft_adj_rmul"

    class StaticArgs(TypedDict):
        op_max_deg: np.int32
        arg_max_deg: np.int32
        op_min_deg: np.int32  # not required
        arg_min_deg: np.int32  # not required

    @staticmethod
    def fallback(
        op_data: Array,
        arg_data: Array,
        width: np.int32,
        depth: np.int32,
        degree_begin: np.ndarray[np.int64.dtype],
        op_max_deg: np.int32,
        arg_max_deg: np.int32,
        op_min_deg: np.int32 = 0,
        arg_min_deg: np.int32 = 0,
    ) -> tuple[Array]:
        op_antipode = _fallback_dense_antipode(op_data, width, depth, degree_begin)
        rmul_antipode = _fallback_ft_adj_lmul(
            op_antipode,
            arg_data,
            depth,
            degree_begin,
            op_max_deg,
            arg_max_deg,
            op_min_deg,
            arg_min_deg,
        )
        rmul = _fallback_dense_antipode(rmul_antipode, width, depth, degree_begin)

        return (rmul,)


class DenseLieToTensor(Operation, DenseOperation):
    fn_name = "lie_to_tensor"

    class StaticArgs(TypedDict):
        l2t_data: np.ndarray[np.float32]
        l2t_indices: np.ndarray[np.int64]
        l2t_indptr: np.ndarray[np.int64]
        l2t_size: np.int64
        scale_factor: Union[None, np.float64]

    def make_static_args(self, kwargs) -> type[TypedDict]:
        arg_basis = self.bases[0]
        tensor_basis = self.bases[1]

        l2t = arg_basis.get_l2t_matrix(self.data_dtype)

        return self.StaticArgs(
            l2t_data=l2t.data,
            l2t_indices=l2t.indices,
            l2t_indptr=l2t.indptr,
            l2t_size=np.int32(tensor_basis.size()),
            **kwargs,
        )

    @staticmethod
    def fallback(
        arg_data: Array,
        width: np.int32,
        depth: np.int32,
        degree_begin: np.ndarray[np.int64.dtype],
        l2t_data: np.ndarray[np.float32],
        l2t_indices: np.ndarray[np.int64],
        l2t_indptr: np.ndarray[np.int64],
        l2t_size: np.int32,
        scale_factor: Union[None, np.float64],
    ) -> tuple[Array]:
        result = csc_matvec(l2t_data, l2t_indices, l2t_indptr, l2t_size, arg_data)
        if scale_factor:
            result = result * scale_factor

        return (result,)


class DenseTensorToLie(Operation, DenseOperation):
    fn_name = "tensor_to_lie"

    class StaticArgs(TypedDict):
        t2l_data: np.ndarray[Union[np.float32, np.float64]]
        t2l_indices: np.ndarray[np.int64]
        t2l_indptr: np.ndarray[np.int64]
        t2l_size: np.int64
        scale_factor: Union[None, np.float64]

    def make_static_args(self, kwargs) -> type[TypedDict]:
        lie_basis = self.bases[1]

        t2l = lie_basis.get_t2l_matrix(self.data_dtype)

        return self.StaticArgs(
            t2l_data=t2l.data,
            t2l_indices=t2l.indices,
            t2l_indptr=t2l.indptr,
            t2l_size=np.int32(lie_basis.size()),
            **kwargs,
        )

    @staticmethod
    def fallback(
        arg_data: Array,
        width: np.int32,
        depth: np.int32,
        degree_begin: np.ndarray[np.int64.dtype],
        t2l_data: np.ndarray[Union[np.float32, np.float64]],
        t2l_indices: np.ndarray[np.int64],
        t2l_indptr: np.ndarray[np.int64],
        t2l_size: np.int32,
        scale_factor: Union[None, np.float64],
    ) -> tuple[Array]:
        result = csc_matvec(t2l_data, t2l_indices, t2l_indptr, t2l_size, arg_data)
        if scale_factor:
            result = result * scale_factor

        return (result,)


## Intermediate operations
class DenseFTExp(Operation, DenseOperation):
    fn_name = "ft_exp"

    class StaticArgs(TypedDict):
        arg_max_deg: np.int32

    @staticmethod
    def fallback(
        arg_data: Array,
        width: np.int32,
        depth: np.int32,
        degree_begin: np.ndarray[np.int64.dtype],
        arg_max_deg: np.int32,
    ) -> tuple[Array]:
        exp = _fallback_dense_ft_exp(arg_data, degree_begin, arg_max_deg)
        return (exp,)


class DenseFTFMExp(Operation, DenseOperation):
    fn_name = "ft_fmexp"

    class StaticArgs(TypedDict):
        out_max_deg: np.int32
        mul_max_deg: np.int32
        exp_max_deg: np.int32
        mul_min_deg: np.int32
        exp_min_deg: np.int32

    @staticmethod
    def fallback(
        multiplier: Array,
        exponent: Array,
        width: np.int32,
        depth: np.int32,
        degree_begin: np.ndarray[np.int64.dtype],
        out_max_deg: np.int32,
        mul_max_deg: np.int32,
        exp_max_deg: np.int32,
        mul_min_deg: np.int32,
        exp_min_deg: np.int32,
    ) -> tuple[Array]:
        # multiplier * exp(exponent)
        exp = _fallback_dense_ft_exp(exponent, degree_begin, exp_max_deg)
        mul = _fallback_dense_ft_mul(
            multiplier,
            exp,
            degree_begin,
            lhs_max_degree=mul_max_deg,
            rhs_max_degree=exp_max_deg,
            out_max_degree=out_max_deg,
            lhs_min_degree=mul_min_deg,
            rhs_min_degree=exp_min_deg,
        )

        return (mul,)


class DenseFTLog(Operation, DenseOperation):
    fn_name = "ft_log"

    class StaticArgs(TypedDict):
        arg_max_deg: np.int32

    @staticmethod
    def fallback(
        arg_data: Array,
        width: np.int32,
        depth: np.int32,
        degree_begin: np.ndarray[np.int64.dtype],
        arg_max_deg: np.int32,
    ) -> tuple[Array]:
        # log(1 + x) = Σ(((-1)^n) * (x^n) / n)
        log = jnp.zeros_like(arg_data)

        for deg in range(arg_max_deg, 0, -1):
            max_level = arg_max_deg - deg + 1

            if deg % 2 == 0:
                log = log.at[0].add(-1 / deg)
            else:
                log = log.at[0].add(1 / deg)

            log = _fallback_dense_ft_mul(
                lhs_data=log,
                rhs_data=arg_data,
                degree_begin=degree_begin,
                out_max_degree=arg_max_deg,
                lhs_min_degree=0,
                lhs_max_degree=max_level,
                rhs_min_degree=1,
                rhs_max_degree=max_level,
            )

        return (log,)


class DenseTensorPairing(Operation, DenseOperation):
    fn_name = "tensor_pairing"

    class StaticArgs(TypedDict):
        functional_max_degree: np.int32
        argument_max_degree: np.int32

    @staticmethod
    def fallback(
        functional_data: Array,
        argument_data: Array,
        width: np.int32,
        depth: np.int32,
        degree_begin: np.ndarray[np.int64.dtype],
        functional_max_degree: np.int32,
        argument_max_degree: np.int32,
    ) -> tuple[Array]:
        common_size = min(
            degree_begin[functional_max_degree + 1],
            degree_begin[argument_max_degree + 1],
        )

        func_data = functional_data[..., :common_size]
        elt_data = argument_data[..., :common_size]

        result = jnp.einsum("...i,...i -> ...", func_data, elt_data)
        return (result,)

    def get_fallback(self) -> Callable[..., Any]:
        return partial(self.fallback, **self.make_ffi_static_args())


# Register all CPU implementations for dense operations
try:
    from ._rpy_jax_internals import cpu_functions
except ImportError as e:
    raise ImportError("RoughPy JAX CPU backend is not installed correctly") from e

Operation.register_all(
    platform="cpu",
    ops=cpu_functions,
    supported_dtypes={"float32", "float64"},
    ffi_register_kwargs={},
)
