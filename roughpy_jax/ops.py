import ctypes
import re

from functools import partial
from typing import ClassVar, Callable, Any, Optional, TypedDict, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jax import Array

try:
    # XLA functions loaded directly from .so rather than python module
    _rpy_jax_internals = ctypes.cdll.LoadLibrary("roughpy_jax/_rpy_jax_internals.so")
except OSError as e:
    _rpy_jax_internals = None
    raise OSError("RoughPy JAX CPU backend is not installed correctly") from e
else:
    # Register CPU functions by looking up expected names in .so
    cpu_func_names = [
        "cpu_dense_ft_fma",
        "cpu_dense_ft_exp",
        "cpu_dense_ft_log",
        "cpu_dense_ft_fmexp",
        "cpu_dense_ft_antipode",
        "cpu_dense_st_fma",
    ]
    for func_name in cpu_func_names:
        func_ptr = getattr(_rpy_jax_internals, func_name)
        jax.ffi.register_ffi_target(
            func_name,
            jax.ffi.pycapsule(func_ptr),
            platform="cpu"
        )


class Operation:
    # Flag to disable the use of accelerated routines in any operation entirely
    # and fall back to using the pure JAX implementations
    #
    # Users should not use this option, it is mostly intended for testing
    # and benchmarking purposes
    no_acceleration: ClassVar[bool] = False

    # The set of all operations defined. This is automatically populated
    # when deriving from this class.
    #
    # Do not interact modify this value.
    __all_operations: ClassVar[dict[tuple[str, str], type[Operation]]] = {}

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

    DynamicParams: ClassVar[type[NamedTuple]]
    StaticParams: ClassVar[type[TypedDict]]

    # Instance parameters, static attributes
    result_shape_dtypes: tuple[jax.ShapeDtypeStruct, ...]
    static_params: type[TypedDict]
    batch_dims: tuple[int, ...]
    data_dtype: jnp.dtype
    ffi_call_args: dict[str, Any]


    @classmethod
    def register(cls, platform: str, name: str, fn_ptr: Any, supported_dtypes: set[jnp.dtype],
                 ffi_register_kwargs: dict[str, Any]):
        jax.ffi.register_ffi_target(name, fn_ptr, platform=platform, **ffi_register_kwargs)

        cls.supported_platforms.add(platform)

        for dtype in supported_dtypes:
            cls.implementations[(platform, str(dtype))] = name

    @classmethod
    def register_all(cls, platform: str, ops: dict[str, Any], supported_dtypes: set[str],
                     ffi_register_kwargs: dict[str, Any]):
        dtypes = {jnp.dtype(tp) for tp in supported_dtypes}
        for op_cls in Operation.__all_operations.values():
            name = f"{platform}_{cls.data_layout}_{op_cls.fn_name}"
            if (fn_ptr := ops.get(name, None)) is not None:
                op_cls.register(platform, name, fn_ptr, dtypes, ffi_register_kwargs)

    @classmethod
    def get_operation(cls, fn_name: str, layout: str = "dense") -> Optional[type[Operation]]:
        key = (fn_name, layout)
        return Operation.__all_operations.get(key, None)

    @classmethod
    def __init_subclass__(cls, **kwargs):
        Operation.__all_operations[cls.fn_name, cls.data_layout] = cls

    def __init__(self, *shape_dtypes, ffi_call_args: Optional[dict[str, Any]], **kwargs):
        self.result_shape_dtypes = shape_dtypes
        self.static_params = self.StaticParams(**kwargs)
        self.data_dtype = shape_dtypes[0].dtype
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
            return partial(fb, **self.static_params, shape_dtypes=self.result_shape_dtypes)

        raise AttributeError("{type(self)} does not name a fallback operation")

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
        def func(*args):
            return jax.ffi.ffi_call(
                name,
                self.result_shape_dtypes,
                **self.ffi_call_args
            )(*args, **self.static_params)

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

    def __call__(self, *data_args):
        fallback = self.get_fallback()

        if self.no_acceleration:
            return fallback(*data_args)

        impls = self.get_implementations()

        return jax.lax.platform_dependent(*data_args, **impls, default=fallback)


def _init_operation(cls):
    if not hasattr(cls, "supported_platforms"):
        cls.supported_platforms = set()

    if not hasattr(cls, "implementations"):
        cls.implementations = {}

    return cls


class DenseOperation:
    data_layout = "dense"


## Basic operations
def _dense_ft_mul_level_accumulator(b_data, c_data, a_deg, basis,
                                    b_min_deg, b_max_deg, c_min_deg, c_max_deg,
                                    dtype, batch_dims):
    db = basis.degree_begin

    out_b = db[a_deg]
    out_e = db[a_deg]
    out_size = out_e - out_b


    acc = jnp.zeros(batch_dims + (out_size,), dtype=dtype)

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



def _get_dense_level_func(b_data, c_data, basis, a_max_deg, b_min_deg, b_max_deg, c_min_deg, c_max_deg) -> Callable[
    [int, Array], Array]:
    def deg_d_update(bdeg, val, *, basis, a_deg):
        db = basis.degree_begin
        cdeg = a_deg - bdeg
        b_level = b_data[db[bdeg]:db[bdeg + 1]]
        c_level = c_data[db[cdeg]:db[cdeg + 1]]
        return val.at[db[a_deg]:db[a_deg + 1]].add(jnp.outer(b_level, c_level).ravel())

    def level_d_func(d, val):
        a_deg = a_max_deg - d
        level_f = partial(deg_d_update, basis=basis, a_deg=a_deg)

        left_deg_start = max(b_min_deg, a_deg - c_max_deg)
        left_deg_end = min(b_max_deg, a_deg - c_min_deg)

        return jax.lax.fori_loop(left_deg_start, left_deg_end + 1, level_f, val, unroll=True)

    return level_d_func


@_init_operation
class DenseFTFma(Operation, DenseOperation):
    fn_name = "ft_fma"

    class DynamicArgs(NamedTuple):
        a_data: Array
        b_data: Array
        c_data: Array
        result: Array

    class StaticArgs(TypedDict, total=True):
        width: np.int32
        depth: np.int32
        degree_begin: np.ndarray[np.intp.dtype]
        out_depth: np.int32
        lhs_depth: np.int32
        rhs_depth: np.int32

    @staticmethod
    def fallback(a_data: Array,
                 b_data: Array,
                 c_data: Array,
                 basis: Any,
                 a_max_deg: np.int32,
                 b_max_deg: np.int32,
                 c_max_deg: np.int32,
                 a_min_deg: np.int32 = 0,
                 b_min_deg: np.int32 = 0,
                 c_min_deg: np.int32 = 0,
                 shape_dtypes: tuple[jax.ShapeDtypeStruct] = ()
                 ) -> Array:
        a_max_deg = min(a_max_deg, b_max_deg + c_max_deg)

        dtype = jnp.result_type(a_data.dtype, b_data.dtype, c_data.dtype)

        batch_dims = a_data.shape[:-1]
        assert batch_dims == b_data.shape[:-1] == c_data.shape[:-1]

        level_gen = partial(
            _dense_ft_mul_level_accumulator,
            b_data=b_data,
            c_data=c_data,
            basis=basis,
            b_min_deg=b_min_deg,
            b_max_deg=b_max_deg,
            c_min_deg=c_min_deg,
            c_max_deg=c_max_deg,
            dtype=dtype,
            batch_dims=batch_dims
        )

        return jnp.concatenate([
            level_gen(i) for i in range(0, a_max_deg)
        ], axis=-1)








@_init_operation
class DenseFTMul(Operation, DenseOperation):
    fn_name = "ft_mul"

    @staticmethod
    def fallback(b_data: Array,
                 c_data: Array,
                 a_max_deg: np.int32,
                 b_max_deg: np.int32,
                 c_max_deg: np.int32,
                 degree_begin: np.ndarray[np.intp.dtype],
                 a_min_deg: np.int32 = 0,
                 b_min_deg: np.int32 = 0,
                 c_min_deg: np.int32 = 0,
                 shape_dtypes: tuple[jax.ShapeDtypeStruct] = ()
                 ) -> Array:
        out_data = jnp.zeros(shape_dtypes[0].shape, dtype=shape_dtypes[0].dtype)

        level_d_func = _get_dense_level_func(degree_begin, b_data, c_data, a_max_deg, b_min_deg, b_max_deg, c_min_deg,
                                             c_max_deg)

        return jax.lax.fori_loop(a_min_deg, a_max_deg, level_d_func, out_data)


@_init_operation
class DenseAntipode(Operation, DenseOperation):
    fn_name = "ft_antipode"

    @staticmethod
    def fallback(
            arg_data: Array,
            width: np.int32,
            depth: np.int32,
            degree_begin: np.ndarray[np.intp.dtype],
            no_sign: bool = False,
            shape_dtypes: tuple[jax.ShapeDtypeStruct] = ()
    ) -> Array:
        db = degree_begin
        out_data = jnp.zeros_like(shape_dtypes[0])

        def transpose_level(i, val):
            sign = 1 if (no_sign or i % 2 == 0) else -1
            level_data = arg_data[db[i]:db[i + 1]].reshape((width,) * i)
            return val.at[db[i]:db[i + 1]].set(sign * jnp.transpose(level_data).ravel())

        return jax.lax.fori_loop(0, depth, transpose_level, out_data, unroll=True)


@_init_operation
class DenseSTFma(Operation, DenseOperation):
    fn_name = "st_fma"


@_init_operation
class DenseFTAdjLeftMul(Operation, DenseOperation):
    fn_name = "ft_adj_lmul"

    @staticmethod
    def fallback(
            op_data: Array,
            arg_data: Array,
            width: np.int32,
            depth: np.int32,
            op_max_deg: np.int32,
            arg_max_deg: np.int32,
            degree_begin: np.ndarray[np.intp.dtype],
            op_min_deg: np.int32 = 0,
            arg_min_deg: np.int32 = 0,
            shape_dtypes: tuple[jax.ShapeDtypeStruct] = ()
    ):
        db = degree_begin

        out_min_deg = arg_min_deg + op_min_deg
        out_max_deg = min(depth, arg_max_deg + op_max_deg)

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

        out_data = jnp.zeros_like(shape_dtypes[0])
        return jax.lax.fori_loop(arg_min_deg, arg_max_deg, arg_deg_loop, out_data)

@_init_operation
class DenseLieToTensor(Operation, DenseOperation):
    fn_name = "lie_to_tensor"


@_init_operation
class DenseTensorToLie(Operation, DenseOperation):
    fn_name = "tensor_to_lie"


## Intermediate operations
@_init_operation
class DenseFTExp(Operation, DenseOperation):
    fn_name = "ft_exp"


@_init_operation
class DenseFTFMExp(Operation, DenseOperation):
    fn_name = "ft_fmexp"


@_init_operation
class DenseFTLog(Operation, DenseOperation):
    fn_name = "ft_log"
