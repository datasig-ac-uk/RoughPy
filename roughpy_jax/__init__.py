import ctypes
import numpy as np
from dataclasses import dataclass
from typing import NamedTuple

try:
    import jax
    import jax.numpy as jnp
    import jax.lax as lax
except ImportError as e:
    raise ImportError("RoughPy JAX requires jax library. For install instructions please refer to https://docs.jax.dev/en/latest/installation.html") from e

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
    ]
    for func_name in cpu_func_names:
        func_ptr = getattr(_rpy_jax_internals, func_name)
        jax.ffi.register_ffi_target(
            func_name,
            jax.ffi.pycapsule(func_ptr),
            platform="cpu"
        )


@dataclass
class TensorBasis:
    """
    Tensor basis with width and depth and array of indexes that specify where
    degree data begins in tensor. If degree_begin is not specified, a default
    free tensor lookup is initialised from width and depth.
    """
    width: int
    depth: int
    degree_begin: jnp.ndarray # FIXME just use np array?

    def __init__(self, width: int, depth: int, degree_begin=None):
        self.width = width
        self.depth = depth

        # Build up a default free tensor degree array if not specified.
        # Equivalent to data[0] = 0; data[i] = 1 + data[i - 1] * width;
        if not degree_begin:
            def degree_begin_fn(last_degree, _):
                new_degree = 1 + last_degree * width
                return new_degree, last_degree

            _, degree_begin = lax.scan(degree_begin_fn, 0, length=depth + 2)
        else:
            # FIXME
            # - check degree_begin type
            # - check size of given degree_begin is at least depth + 2
            pass

        self.degree_begin = degree_begin

    def size(self):
        return self.degree_begin[self.depth + 1]


class DenseFreeTensor(NamedTuple):
    """
    Dense free tensor class built from basis and associated ndarray of data.
    """
    data: jnp.ndarray
    basis: TensorBasis


"""
Free tensor alias. Tensors are assumed to be dense without prefix
"""
FreeTensor = DenseFreeTensor


def _check_basis_compat(out_basis: TensorBasis, *other_bases: TensorBasis):
    out_width = out_basis.width

    for i, basis in enumerate(other_bases):
        if basis.width != out_width:
            raise ValueError(f"Incompatible width for basis {i}")


def _check_tensor_data_type(ft: FreeTensor, err_msg: str):
    if ft.data.dtype != jnp.float32:
        raise ValueError(err_msg)


def ft_fma(a: FreeTensor, b: FreeTensor, c: FreeTensor) -> FreeTensor:
    """
    Free tensor fused multiply-add

    This function is equivalent to `b * c + a`.
    Currently only floating point scalars (np.float32) are supported.
    The basis is taken from `c`.

    :param a: addition operand
    :param b: left-hand multiply operand
    :param c: right-hand multiple operand
    :return: result
    """
    _check_tensor_data_type(a, "ft_fma a expecting float32 array data")
    _check_tensor_data_type(b, "ft_fma b expecting float32 array data")
    _check_tensor_data_type(c, "ft_fma c expecting float32 array data")
    _check_basis_compat(a.basis, b.basis, c.basis)

    # Use same basis convention as ft_fma in roughpy/compute
    basis = c.basis
    out_depth = c.basis.depth
    lhs_depth = -1
    rhs_depth = -1

    call = jax.ffi.ffi_call(
        "cpu_dense_ft_fma",
        jax.ShapeDtypeStruct(c.data.shape, c.data.dtype)
    )

    out_data = call(
        basis.degree_begin,
        a.data,
        b.data,
        c.data,
        width=np.int32(basis.width),
        depth=np.int32(basis.depth),
        out_depth=np.int32(out_depth),
        lhs_depth=np.int32(lhs_depth),
        rhs_depth=np.int32(rhs_depth)
    )

    return FreeTensor(out_data, basis)


def ft_mul(a: FreeTensor, b: FreeTensor) -> FreeTensor:
    """
    Free tensor multiply

    This function is equivalent to `a * b`.
    Currently only floating point scalars (np.float32) are supported.
    The basis is taken from `b`.

    :param a: left-hand multiply operand
    :param b: right-hand multiple operand
    :return: result
    """
    _check_tensor_data_type(a, "ft_mul a expecting float32 array data")
    _check_tensor_data_type(b, "ft_mul b expecting float32 array data")
    _check_basis_compat(a.basis, b.basis)

    zero_add_data = jnp.zeros_like(a.data)

    # Use same basis convention as ft_mul in roughpy/compute
    basis = b.basis
    out_depth = b.basis.depth
    lhs_depth = -1
    rhs_depth = b.basis.depth

    call = jax.ffi.ffi_call(
        "cpu_dense_ft_fma",
        jax.ShapeDtypeStruct(b.data.shape, b.data.dtype)
    )

    out_data = call(
        basis.degree_begin,
        zero_add_data,
        a.data,
        b.data,
        width=np.int32(basis.width),
        depth=np.int32(basis.depth),
        out_depth=np.int32(out_depth),
        lhs_depth=np.int32(lhs_depth),
        rhs_depth=np.int32(rhs_depth)
    )

    return FreeTensor(out_data, basis)


# FIXME review: should this have ft_ prefix? compute version does not
def ft_antipode(a: FreeTensor) -> FreeTensor:
    """
    Antipode of a free tensor

    :param a: argument
    :return: new tensor with antipode of `a`
    """
    _check_tensor_data_type(a, "antipode a expecting float32 array data")

    out_basis = a.basis

    call = jax.ffi.ffi_call(
        "cpu_dense_ft_antipode",
        jax.ShapeDtypeStruct(a.data.shape, a.data.dtype)
    )

    out_data = call(
        out_basis.degree_begin,
        a.data,
        width=np.int32(out_basis.width),
        depth=np.int32(out_basis.depth),
        arg_depth=np.int32(a.basis.depth)
    )

    return FreeTensor(out_data, out_basis)


def ft_exp(x: FreeTensor, out_basis: TensorBasis | None = None) -> FreeTensor:
    """
    Free tensor exponent

    This function is equivalent to `e ^ x`.
    Currently only floating point scalars (np.float32) are supported.
    If `out_basis` is not specified, the same basis as `x` is used.

    :param x: argument
    :param out_basis: optional output basis.
    :return: tensor exponential of `x`
    """
    _check_tensor_data_type(x, "ft_exp x expecting float32 array data")

    out_basis = out_basis or x.basis

    call = jax.ffi.ffi_call(
        "cpu_dense_ft_exp",
        jax.ShapeDtypeStruct(x.data.shape, x.data.dtype)
    )

    out_data = call(
        out_basis.degree_begin,
        x.data,
        width=np.int32(out_basis.width),
        depth=np.int32(out_basis.depth),
        arg_depth=np.int32(x.basis.depth)
    )

    return FreeTensor(out_data, out_basis)


def ft_log(x: FreeTensor, out_basis: TensorBasis | None = None) -> FreeTensor:
    """
    Free tensor logarithm

    This function is equivalent to `log(x)`.
    Currently only floating point scalars (np.float32) are supported.
    If `out_basis` is not specified, the same basis as `x` is used.

    :param x: argument
    :param out_basis: optional output basis.
    :return: tensor logarithm of `x`
    """
    _check_tensor_data_type(x, "ft_log x expecting float32 array data")

    out_basis = out_basis or x.basis

    call = jax.ffi.ffi_call(
        "cpu_dense_ft_log",
        jax.ShapeDtypeStruct(x.data.shape, x.data.dtype)
    )

    out_data = call(
        out_basis.degree_begin,
        x.data,
        width=np.int32(out_basis.width),
        depth=np.int32(out_basis.depth),
        arg_depth=np.int32(x.basis.depth)
    )

    return FreeTensor(out_data, out_basis)


def ft_fmexp(multiplier: FreeTensor, exponent: FreeTensor, out_basis: TensorBasis | None = None) -> FreeTensor:
    """
    Free tensor fused multiply-exponential

    This function is equivalent to `a * exp(x)` where `a` is `multiplier` and `x` is `exponent`.
    Currently only floating point scalars (np.float32) are supported.
    If `out_basis` is not specified, the same basis as `multiplier` is used.

    :param multiplier: Multiplier free tensor
    :param exponent: Free tensor to exponential
    :param out_basis: Optional output basis. If not specified, the same basis as `multiplier` is used.
    :return: Resulting fused multiply-exponential of `multiplier` and `exponent`
    """
    _check_tensor_data_type(multiplier, "ft_fmexp multiplier expecting float32 array data")
    _check_tensor_data_type(exponent, "ft_fmexp exponent expecting float32 array data")

    out_basis = out_basis or multiplier.basis
    _check_basis_compat(out_basis, multiplier.basis, exponent.basis)

    basis = multiplier.basis
    out_depth = multiplier.basis.depth
    mul_depth = -1
    exp_depth = -1

    call = jax.ffi.ffi_call(
        "cpu_dense_ft_fmexp",
        jax.ShapeDtypeStruct(multiplier.data.shape, multiplier.data.dtype)
    )

    out_data = call(
        basis.degree_begin,
        multiplier.data,
        exponent.data,
        width=np.int32(basis.width),
        depth=np.int32(basis.depth),
        out_depth=np.int32(out_depth),
        mul_depth=np.int32(mul_depth),
        exp_depth=np.int32(exp_depth)
    )

    return FreeTensor(out_data, basis)
