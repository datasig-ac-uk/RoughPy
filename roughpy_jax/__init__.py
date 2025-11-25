import jax
import jax.numpy as jnp
import numpy as np

from .ffi import *
from .basis import TensorBasis, LieBasis
from .tensor import FreeTensor, ShuffleTensor, Lie

# FIXME temporary import for l2t and l2t
import _rpy_compute_internals


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
    Supports float 32 or 64 but all data buffers must have matching type.
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
    Supports float 32 or 64 but all data buffers must have matching type.
    The basis is taken from `b`.

    :param a: left-hand multiply operand
    :param b: right-hand multiple operand
    :return: result
    """
    _check_tensor_dtype(a, b)
    _check_basis_compat(a.basis, b.basis)

    # Zero data for underlying fma
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


def st_fma(a: ShuffleTensor, b: ShuffleTensor, c: ShuffleTensor) -> ShuffleTensor:
    """
    Shuffle tensor fused multiply-add

    This function is equivalent to `b * c + a`.
    Supports float 32 or 64 but all data buffers must have matching type.
    The result basis is taken from `a`.

    :param a: input and first operand
    :param b: left-hand operand
    :param c: right-hand operand
    :return: shuffle fused multiply-add
    """
    _check_tensor_dtype(a, b)
    _check_basis_compat(a.basis, b.basis, c.basis)

    call = jax.ffi.ffi_call(
        "cpu_dense_st_fma",
        jax.ShapeDtypeStruct(a.data.shape, a.data.dtype)
    )

    out_data = call(
        a.data,
        b.data,
        c.data,
        width=np.int32(a.width),
        depth=np.int32(a.depth),
        arg_depth=np.int32(a.basis.depth),
        degree_begin=a.degree_begin
    )

    return ShuffleTensor(out_data, a.basis)


def st_mul(lhs: ShuffleTensor, rhs: ShuffleTensor) -> ShuffleTensor:
    """
    Shuffle tensor product

    This function is equivalent to `lhs & rhs`.
    Supports float 32 or 64 but all data buffers must have matching type.
    The result basis is taken from `lhs`.

    :param lhs: left-hand operand
    :param rhs: right-hand operand
    :return: the shuffle product of lhs and rhs
    """
    _check_tensor_dtype(lhs, rhs)
    _check_basis_compat(lhs.basis, rhs.basis)

    # Zero data for underlying fma
    zero_add_data = jnp.zeros_like(lhs.data)

    call = jax.ffi.ffi_call(
        "cpu_dense_st_fma",
        jax.ShapeDtypeStruct(lhs.data.shape, lhs.data.dtype)
    )

    out_data = call(
        zero_add_data,
        lhs.data,
        rhs.data,
        width=np.int32(lhs.basis.width),
        depth=np.int32(lhs.basis.depth),
        arg_depth=np.int32(lhs.basis.depth),
        degree_begin=lhs.basis.degree_begin
    )

    return ShuffleTensor(out_data, lhs.basis)


def ft_exp(x: FreeTensor, out_basis: TensorBasis | None = None) -> FreeTensor:
    """
    Free tensor exponent

    This function is equivalent to `e ^ x`.
    Supports float 32 or 64 but all data buffers must have matching type.
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
    Supports float 32 or 64 but all data buffers must have matching type.
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
    Supports float 32 or 64 but all data buffers must have matching type.
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


def lie_to_tensor(arg: Lie, tensor_basis: TensorBasis | None = None, scale_factor=None) -> FreeTensor:
    """
    Compute the embedding of a Lie algebra element as a free tensor.

    :param arg: Lie to embed into the tensor algebra
    :param tensor_basis: optional tensor basis to embed. Must have the same width as the Lie basis.
    :return: new FreeTensor containing the embedding of "arg"
    """
    l2t = arg.basis.get_l2t_matrix(arg.data.dtype)

    tensor_basis = tensor_basis or TensorBasis(arg.basis.width, arg.basis.depth)

    # FIXME placeholder code, running with compute code before migration to JAX
    result = np.zeros((*arg.data.shape[:-1], tensor_basis.size()), dtype=arg.data.dtype)
    _rpy_compute_internals.dense_lie_to_tensor(result, arg.data, l2t, arg.basis, tensor_basis, scale_factor=arg.data.dtype.type(scale_factor) if scale_factor is not None else None)

    return FreeTensor(result, tensor_basis)


def tensor_to_lie(arg: FreeTensor, lie_basis: LieBasis | None = None, scale_factor=None) -> Lie:
    """
    Project a free tensor onto the embedding of the Lie algebra in the tensor algebra.

    :param arg:
    :param lie_basis:
    :return:
    """
    lie_basis = lie_basis or LieBasis(arg.basis.width, arg.basis.depth)
    l2t = lie_basis.get_t2l_matrix(arg.data.dtype)

    # FIXME placeholder code, running with compute code before migration to JAX
    result = np.zeros((*arg.data.shape[:-1], lie_basis.size()), dtype=arg.data.dtype)
    _rpy_compute_internals.dense_tensor_to_lie(result, arg.data, l2t, lie_basis, arg.basis, scale_factor=arg.data.dtype.type(scale_factor) if scale_factor is not None else None)

    return Lie(result, lie_basis)
