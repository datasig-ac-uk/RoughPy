import ctypes
import numpy as np
from typing import NamedTuple

try:
    import jax
    import jax.numpy as jnp
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


# FIXME review point: neatly load compute module from roughpy_jax dir
import sys
sys.path.append('roughpy/compute')
import _rpy_compute_internals


class TensorBasis(_rpy_compute_internals.TensorBasis):
    pass


@jax.tree_util.register_pytree_node_class
class DenseFreeTensor(NamedTuple):
    """
    Dense free tensor class built from basis and associated ndarray of data.
    """
    data: jnp.ndarray
    basis: TensorBasis

    def tree_flatten(self):
        # Flatten dynamic data with static basis
        return (self.data,), self.basis

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # Reconstruct from static and dynamic data
        (data,) = children
        basis = aux_data
        return cls(data, basis)


"""
Free tensor alias. Tensors are assumed to be dense without prefix
"""
FreeTensor = DenseFreeTensor


def _check_basis_compat(first_basis: TensorBasis, *other_bases: TensorBasis):
    for i, basis in enumerate(other_bases):
        if basis.width != first_basis.width:
            raise ValueError(f"Incompatible width between basis 0 and basis {i + 1}")


def _check_tensor_dtype(first_tensor: FreeTensor, *other_tensors: FreeTensor):
    for i, ft in enumerate([first_tensor] + list(other_tensors)):
        if ft.data.dtype != jnp.float32:
            if ft.data.dtype != jnp.float64:
                raise ValueError(f"Expecting jnp.float32 or jnp.float64 array for tensor {i}")

    for i, tensor in enumerate(other_tensors):
        if tensor.data.dtype != first_tensor.data.dtype:
            raise ValueError(f"Incompatible dtype between tensor 0 and tensor {i + 1}")


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
    _check_tensor_dtype(a, b, c)
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
        a.data,
        b.data,
        c.data,
        width=np.int32(basis.width),
        depth=np.int32(basis.depth),
        out_depth=np.int32(out_depth),
        lhs_depth=np.int32(lhs_depth),
        rhs_depth=np.int32(rhs_depth),
        degree_begin=basis.degree_begin
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
    _check_tensor_dtype(a, b)
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
        zero_add_data,
        a.data,
        b.data,
        width=np.int32(basis.width),
        depth=np.int32(basis.depth),
        out_depth=np.int32(out_depth),
        lhs_depth=np.int32(lhs_depth),
        rhs_depth=np.int32(rhs_depth),
        degree_begin=basis.degree_begin
    )

    return FreeTensor(out_data, basis)


def antipode(a: FreeTensor) -> FreeTensor:
    """
    Antipode of a free tensor

    :param a: argument
    :return: new tensor with antipode of `a`
    """
    _check_tensor_dtype(a)

    out_basis = a.basis

    call = jax.ffi.ffi_call(
        "cpu_dense_ft_antipode",
        jax.ShapeDtypeStruct(a.data.shape, a.data.dtype)
    )

    out_data = call(
        a.data,
        width=np.int32(out_basis.width),
        depth=np.int32(out_basis.depth),
        arg_depth=np.int32(a.basis.depth),
        degree_begin=out_basis.degree_begin
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
    _check_tensor_dtype(x)

    out_basis = out_basis or x.basis

    call = jax.ffi.ffi_call(
        "cpu_dense_ft_exp",
        jax.ShapeDtypeStruct(x.data.shape, x.data.dtype)
    )

    out_data = call(
        x.data,
        width=np.int32(out_basis.width),
        depth=np.int32(out_basis.depth),
        arg_depth=np.int32(x.basis.depth),
        degree_begin=out_basis.degree_begin
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
    _check_tensor_dtype(x)

    out_basis = out_basis or x.basis

    call = jax.ffi.ffi_call(
        "cpu_dense_ft_log",
        jax.ShapeDtypeStruct(x.data.shape, x.data.dtype)
    )

    out_data = call(
        x.data,
        width=np.int32(out_basis.width),
        depth=np.int32(out_basis.depth),
        arg_depth=np.int32(x.basis.depth),
        degree_begin=out_basis.degree_begin
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
    _check_tensor_dtype(multiplier, exponent)

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
        multiplier.data,
        exponent.data,
        width=np.int32(basis.width),
        depth=np.int32(basis.depth),
        out_depth=np.int32(out_depth),
        mul_depth=np.int32(mul_depth),
        exp_depth=np.int32(exp_depth),
        degree_begin=out_basis.degree_begin
    )

    return FreeTensor(out_data, basis)
