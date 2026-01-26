import typing
import collections.abc as cabc

from functools import partial
from typing import ClassVar, Callable, Any, Optional, TypedDict, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

from jax import Array

try:
    # Import using python module syntax to ensure proper loading of shared library
    from ._rpy_jax_internals import cpu_functions
except ImportError as e:
    _rpy_jax_internals = None
    raise ImportError(
        "RoughPy JAX CPU backend is not installed correctly"
    ) from e


class BasisLike(typing.Protocol, cabc.Hashable):
    width: np.int32
    depth: np.int32
    degree_begin: np.ndarray[np.int64.dtype]

    def size(self) -> int:
        ...



class EmptyStaticArgs(TypedDict):
    ...


OperationT = TypeVar("OperationT")


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
    supported_platforms: ClassVar[set[str]] # = set() FIXME

    # The set of implementations operations for each operation. This is a mapping
    # from platform, dtype tuple to function name. The function name should
    # follow the format described above.
    implementations: ClassVar[dict[tuple[str, str], str]] # = {} FIXME

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
    def register(cls, platform: str, name: str, fn_ptr: Any, supported_dtypes: set[jnp.dtype],
                 ffi_register_kwargs: dict[str, Any]):
        cls.supported_platforms.add(platform)

        for dtype in supported_dtypes:
            key = (platform, str(dtype))
            if key in cls.implementations:
                raise ValueError(f"Implementation {key} already registered for operation {cls.__name__}")
            cls.implementations[key] = name

        jax.ffi.register_ffi_target(name, fn_ptr, platform=platform, **ffi_register_kwargs)

    @classmethod
    def register_all(cls, platform: str, ops: dict[str, Any], supported_dtypes: set[str],
                     ffi_register_kwargs: dict[str, Any]):
        dtypes = {jnp.dtype(tp) for tp in supported_dtypes}
        for op_cls in Operation.__all_operations.values():
            name = f"{platform}_{cls.data_layout}_{op_cls.fn_name}"
            if (fn_ptr := ops.get(name, None)) is not None:
                op_cls.register(platform, name, fn_ptr, dtypes, ffi_register_kwargs)

    @classmethod
    def get_operation(cls, fn_name: str, layout: str = "dense") -> Optional[type[OperationT]]:
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
        return (jax.ShapeDtypeStruct(
            batch_dims + (basis.size(),),
            dtype
        ),)

    @classmethod
    def get_result_basis(cls, bases: tuple[BasisLike, ...], preferred_basis) -> BasisLike:
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
            raise ValueError(f"mismatched width on basis 0, expected {preferred_basis.width} but got {choice.width}")

        for i, basis in enumerate(other, start=1):
            if basis.width != choice.width:
                raise ValueError(f"mismatched width on basis {i}, expected {choice.width} but got {basis.width}")

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

    def __init__(self, bases, dtype, batch_dims, ffi_call_args: Optional[dict[str, Any]] = None,
                 specific_basis: Optional[BasisLike] = None, **kwargs):
        self.basis = basis = self.get_result_basis(bases, specific_basis)
        self.bases = bases
        self.data_dtype = dtype
        self.batch_dims = batch_dims
        self.static_args = self.StaticArgs(**kwargs)

        self.result_shape_dtypes = self.make_result_dtypes(basis, dtype, batch_dims)
        self.ffi_call_args = self.default_ffi_call_args | (ffi_call_args or {})

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
            return partial(fb, **self.make_ffi_static_args())

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
            **self.static_args
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
                name,
                self.result_shape_dtypes,
                **self.ffi_call_args
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
            platform: self.make_ffi_call(impl_name) for platform in self.supported_platforms
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
def _dense_ft_mul_level_accumulator(b_data, c_data, a_deg, basis,
                                    b_min_deg, b_max_deg, c_min_deg, c_max_deg):
    db = basis.degree_begin

    out_b = db[a_deg]
    out_e = db[a_deg]
    out_size = out_e - out_b

    acc = jnp.zeros(b_data.shape[:-1] + (out_size,), dtype=b_data.dtype)

    b_deg_b = max(b_min_deg, a_deg - c_max_deg)
    b_deg_e = min(b_max_deg, a_deg - c_min_deg) + 1

    for b_deg in range(b_deg_b, b_deg_e):
        c_deg = a_deg - b_deg
        b_b = db[b_deg]
        b_e = db[b_deg+1]
        c_b = db[c_deg]
        c_e = db[c_deg+1]

        b_level = b_data[..., b_b:b_e]
        c_level = c_data[..., c_b:c_e]

        acc = acc + jnp.tensordot(b_level, c_level, axes=0).reshape(-1)

    return acc


class DenseFTFma(Operation, DenseOperation):
    fn_name = "ft_fma"


    class StaticArgs(TypedDict):
        a_max_deg: np.int32
        b_max_deg: np.int32
        c_max_deg: np.int32
        b_min_deg: np.int32 # not required
        c_min_deg: np.int32 # not required



    @staticmethod
    def fallback(a_data: Array,
                 b_data: Array,
                 c_data: Array,
                 width: np.int32, # FIXME found during merge, fallback code was still using basis and not taking ffi args
                 depth: np.int32,
                 degree_begin: np.ndarray[np.int64.dtype],
                 a_max_deg: np.int32,
                 b_max_deg: np.int32,
                 c_max_deg: np.int32,
                 b_min_deg: np.int32 = 0,
                 c_min_deg: np.int32 = 0,
                 ) -> tuple[Array, ...]:
        level_gen = partial(
            _dense_ft_mul_level_accumulator,
            b_data=b_data,
            c_data=c_data,
            basis=basis,
            b_min_deg=b_min_deg,
            b_max_deg=b_max_deg,
            c_min_deg=c_min_deg,
            c_max_deg=c_max_deg,
        )

        mul = jnp.concatenate([level_gen(i) for i in range(0, a_max_deg)], axis=-1)

        return (a_data + mul,)


class DenseFTMul(Operation, DenseOperation):
    fn_name = "ft_mul"

    class StaticArgs(TypedDict):
        out_max_deg: np.int32
        lhs_max_deg: np.int32
        rhs_max_deg: np.int32

    @staticmethod
    def fallback(b_data: Array,
                 c_data: Array,
                 basis: Any,
                 out_max_deg: np.int32,
                 lhs_max_deg: np.int32,
                 rhs_max_deg: np.int32,
                 lhs_min_deg: np.int32 = 0,
                 rhs_min_deg: np.int32 = 0,
                 ) -> tuple[Array]:

        batch_dims = b_data.shape[:-1]
        assert batch_dims == c_data.shape[:-1]

        level_gen = partial(
            _dense_ft_mul_level_accumulator,
            b_data=b_data,
            c_data=c_data,
            basis=basis,
            b_min_deg=lhs_min_deg,
            b_max_deg=lhs_max_deg,
            c_min_deg=rhs_min_deg,
            c_max_deg=rhs_max_deg,
        )

        return (jnp.concatenate([
            level_gen(i) for i in range(0, out_max_deg)
        ], axis=-1),)




class DenseAntipode(Operation, DenseOperation):
    fn_name = "ft_antipode"

    class StaticArgs(TypedDict):
        no_sign: bool

    @staticmethod
    def fallback(
            arg_data: Array,
            basis: Any,
            no_sign: bool = False,
    ) -> tuple[Array]:
        db = basis.degree_begin

        def transpose_level(i):
            sign = 1 if (no_sign or i % 2 == 0) else -1
            level_data = arg_data[db[i]:db[i + 1]].reshape((basis.width,) * i)
            return sign * jnp.transpose(level_data).reshape(-1)

        data = jnp.concatenate(
            [transpose_level(i) for i in range(basis.depth + 1)],
            axis=-1
        )

        return (data, )

class DenseSTFma(Operation, DenseOperation):
    fn_name = "st_fma"

    class StaticArgs(TypedDict):
        a_max_deg: np.int32
        b_max_deg: np.int32
        c_max_deg: np.int32
        b_min_deg: np.int32 # not required
        c_min_deg: np.int32 # not required

    @staticmethod
    def fallback(
            a_data: Array,
            b_data: Array,
            c_data: Array,
            basis: BasisLike,
            a_max_deg: np.int32,
            b_max_deg: np.int32,
            c_max_deg: np.int32,
            b_min_deg: np.int32 = 0,
            c_min_deg: np.int32 = 0,
    ) -> tuple[Array]:
        raise NotImplementedError

class DenseSTMul(Operation, DenseOperation):
    fn_name = "st_mul"

    class StaticArgs(TypedDict):
        lhs_max_deg: np.int32
        rhs_max_deg: np.int32
        lhs_min_deg: np.int32 # not required
        rhs_min_deg: np.int32 # not required

    @staticmethod
    def fallback(
            b_data: Array,
            c_data: Array,
            basis: BasisLike,
            out_max_deg: np.int32,
            lhs_max_deg: np.int32,
            rhs_max_deg: np.int32,
            lhs_min_deg: np.int32 = 0,
            rhs_min_deg: np.int32 = 0,
    ) -> tuple[Array]:
        raise NotImplementedError




class DenseFTAdjLeftMul(Operation, DenseOperation):
    fn_name = "ft_adj_lmul"

    class StaticArgs(TypedDict):
        op_max_deg: np.int32
        arg_max_deg: np.int32
        op_min_deg: np.int32  # not required
        arg_min_deg: np.int32 # not required


    @staticmethod
    def fallback(
            op_data: Array,
            arg_data: Array,
            basis: BasisLike,
            op_max_deg: np.int32,
            arg_max_deg: np.int32,
            op_min_deg: np.int32 = 0,
            arg_min_deg: np.int32 = 0,
    ):
        db = basis.degree_begin
        out_max_deg = basis.depth
        out_min_deg = 0

        batch_dims = arg_data.shape[:-1]

        out_data = jnp.zeros(batch_dims + (db[basis.depth],), dtype=arg_data.dtype)


        def dsize(degree):
            return db[degree+1] - db[degree]

        def inner_fn(op_idx, val, *, op_deg, out_deg):
            arg_deg = op_deg + out_deg
            out_size = dsize(out_deg)
            offset = db[arg_deg] + op_idx * out_size
            op_val = op_data[db[op_deg] + op_idx]

            arg_level = arg_data[offset: offset + out_size]
            return val.at[db[out_deg]:db[out_deg+1]].add(op_val * arg_level)

        def out_deg_loop(out_deg, val, *, arg_deg):
            op_deg = arg_deg - out_deg

            fn = partial(inner_fn, op_deg = op_deg, out_deg = out_deg)
            return jax.lax.fori_loop(0, dsize(op_deg), fn, val)

        def arg_deg_loop(d, val):
            arg_deg = arg_max_deg - d

            odeg_start = max(arg_deg - op_max_deg, out_min_deg)
            odeg_end = min(arg_deg - op_min_deg, out_max_deg)

            return jax.lax.fori_loop(odeg_start, odeg_end, partial(out_deg_loop, arg_deg=arg_deg), val)

        return jax.lax.fori_loop(arg_min_deg, arg_max_deg, arg_deg_loop, out_data)

class DenseLieToTensor(Operation, DenseOperation):
    fn_name = "lie_to_tensor"


class DenseTensorToLie(Operation, DenseOperation):
    fn_name = "tensor_to_lie"


## Intermediate operations
class DenseFTExp(Operation, DenseOperation):
    fn_name = "ft_exp"


class DenseFTFMExp(Operation, DenseOperation):
    fn_name = "ft_fmexp"


class DenseFTLog(Operation, DenseOperation):
    fn_name = "ft_log"


# FIXME work in progress code to run fallback code if no cpu code registered
ADD_STANDARD_IMPLS = True

if ADD_STANDARD_IMPLS:
    standard_cpu_ops = [
        DenseFTFma,
        DenseFTMul,
        DenseAntipode,
        DenseSTFma,
        DenseSTMul,
        DenseFTAdjLeftMul,
        DenseFTExp,
        DenseFTFMExp,
        DenseFTLog
        # FIXME incomplete:
        # DenseLieToTensor
        # DenseTensorToLie
    ]

    for op in standard_cpu_ops:
        cpu_function_name = f"cpu_dense_{op.fn_name}"
        op.register(
            "cpu",
            op.fn_name,
            cpu_functions[cpu_function_name],
            {"float32", "float64"},
            {}
        )

    # FIXME methods that have yet to be converted to new ops framework
    legacy_cpu_functions = [
        "cpu_dense_ft_exp",
        "cpu_dense_ft_log",
        "cpu_dense_ft_fmexp",
        "cpu_dense_ft_adj_lmul",
        "cpu_dense_ft_adj_rmul"
    ]

    for fn in legacy_cpu_functions:
        jax.ffi.register_ffi_target(fn, cpu_functions[fn], platform="cpu")
