from collections.abc import Callable
from dataclasses import dataclass
from functools import partialmethod
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

from roughpy_jax.ops import Operation

from .bases import (
    LieBasis,
    TensorBasis,
    check_basis_compat,
    to_lie_basis,
)
from .compressed import csr_matvec

T = TypeVar("T")
AlgebraT = TypeVar("AlgebraT")
FreeTensorT = TypeVar("FreeTensorT")
ShuffleTensorT = TypeVar("ShuffleTensorT")
LieT = TypeVar("LieT")


def _algebra_add(
    a: AlgebraT, b: AlgebraT, *, impl: Callable[[jax.Array, ...], jax.Array]
) -> AlgebraT:
    cls = type(a)

    if not issubclass(type(b), cls):
        return NotImplemented

    basis = a.basis
    # TODO: check basis and batching dims etc for better messages or adjusting truncation depth

    result_data = impl(a.data, b.data)

    return cls(result_data, basis)


def _broadcast_to_batch_shape(
    data: jax.Array, batch_shape: tuple[int, ...], *, core_dims=1
) -> jax.Array:
    data_shape = data.shape

    ds_len = len(data_shape)
    bs_len = len(batch_shape)

    if ds_len <= bs_len and data_shape == batch_shape[:ds_len]:
        new_shape = data_shape + (1,) * (bs_len - ds_len) + (1,) * core_dims
    elif (
        ds_len == bs_len + 1 and data_shape[:-1] == batch_shape and data_shape[-1] == 1
    ):
        new_shape = data_shape
    else:
        raise ValueError(
            f"batch shape {batch_shape} is incompatible with data shape {data_shape}"
        )

    return jnp.reshape(data, new_shape)


def _algebra_scalar_multiply(a: AlgebraT, s: jax.typing.ArrayLike) -> AlgebraT:
    cls = type(a)
    basis = a.basis

    scalar = jnp.asarray(s)
    ext_scalar = _broadcast_to_batch_shape(scalar, a.batch_shape)

    result_data = jnp.multiply(a.data, ext_scalar)

    return cls(result_data, basis)


def _algebra___array__(self, dtype=None, copy=None) -> np.ndarray:
    return self.data.__array__(dtype=dtype, copy=copy)


def _redepth_data(data: jax.Array, new_alg_dim: int) -> jax.Array:
    shape = data.shape

    if new_alg_dim < shape[-1]:
        return data[..., :new_alg_dim]

    pad_dims = [(0, 0)] * (len(shape) - 1) + [(0, new_alg_dim - shape[-1])]
    return jnp.pad(data, pad_dims)


def _algebra_change_depth(algebra: AlgebraT, new_depth: int) -> AlgebraT:
    if new_depth == algebra.basis.depth:
        return algebra

    algebra_cls = type(algebra)
    basis_cls = type(algebra.basis)

    new_basis = basis_cls(algebra.basis.width, new_depth)

    new_size = new_basis.size()

    return algebra_cls(_redepth_data(algebra.data, new_size), new_basis)


def _tensor_dataclass(cls):
    """
    Combined decorator for roughpy_jax tensor objects

    Registers dataclass and JAX data class with dynamic data and static basis
    """
    cls = dataclass(cls)

    cls.__array__ = _algebra___array__

    cls.__add__ = partialmethod(_algebra_add, impl=jnp.add)
    cls.__sub__ = partialmethod(_algebra_add, impl=jnp.subtract)

    cls.__radd__ = lambda x, y: _algebra_add(y, x, impl=jnp.add)
    cls.__rsub__ = lambda x, y: _algebra_add(x, y, impl=jnp.subtract)

    def _mul_impl(self, other):
        if isinstance(other, (jax.Array, np.ndarray, np.generic, float, int)):
            return _algebra_scalar_multiply(self, other)
        return NotImplemented

    cls.__mul__ = _mul_impl

    def _rmul_impl(self, other):
        if isinstance(other, (jax.Array, np.ndarray, np.generic, float, int)):
            return _algebra_scalar_multiply(self, other)
        return NotImplemented

    cls.__rmul__ = _rmul_impl

    def _div_impl(self, other):
        if isinstance(other, (jax.Array, np.ndarray, np.generic, float, int)):
            return _algebra_scalar_multiply(self, 1.0 / other)
        return NotImplemented

    cls.__truediv__ = _div_impl

    cls.change_depth = _algebra_change_depth

    return jax.tree_util.register_dataclass(
        cls, data_fields=["data"], meta_fields=["basis"]
    )


@_tensor_dataclass
class DenseFreeTensor:
    """
    Dense free tensor class built from basis and associated ndarray of data.
    """

    data: jnp.ndarray
    basis: TensorBasis

    @property
    def batch_shape(self):
        return self.data.shape[:-1]


@_tensor_dataclass
class DenseShuffleTensor:
    """
    Dense shuffle tensor class built from basis and associated ndarray of data.
    """

    data: jnp.ndarray
    basis: TensorBasis

    @property
    def batch_shape(self):
        return self.data.shape[:-1]


@_tensor_dataclass
class DenseLie:
    data: jnp.ndarray
    basis: LieBasis

    @property
    def batch_shape(self):
        return self.data.shape[:-1]


"""
Tensor aliases. Tensors are assumed to be dense without prefix

TODO: These should be replaced by type aliases later, and we should
      look at the types of the inputs to determine the correct type
      for the output in each case. For now, we assume everything is
      dense, but we should nonetheless do this replacement soon.
"""
FreeTensor = DenseFreeTensor
ShuffleTensor = DenseShuffleTensor
Lie = DenseLie


def _check_tensor_dtype(first_tensor: FreeTensor, *other_tensors: FreeTensor):
    for i, ft in enumerate((first_tensor, *list(other_tensors))):
        if ft.data.dtype != jnp.float32:
            if ft.data.dtype != jnp.float64:
                raise ValueError(
                    f"Expecting jnp.float32 or jnp.float64 array for tensor {i}"
                )

    for i, tensor in enumerate(other_tensors):
        if tensor.data.dtype != first_tensor.data.dtype:
            raise ValueError(f"Incompatible dtype between tensor 0 and tensor {i + 1}")


def _get_and_check_batch_dims(*arrays, core_dims=1):
    if not arrays:
        raise ValueError("expected at least one array")

    first, *rem = arrays

    n_dims = len(first.shape)
    if n_dims < core_dims:
        raise ValueError(
            f"array at index 0 has wrong number of dimensions, expected at least {core_dims}"
        )

    batch_dims = first.shape[:-core_dims]
    n_batch_dims = len(batch_dims)

    for i, arr in enumerate(rem, start=1):
        if len(arr.shape) != n_dims:
            raise ValueError(
                f"mismatched number of dimensions at index {i}: "
                f"expected {n_dims} but got {len(arr.shape)}"
            )

        if arr.shape[:n_batch_dims] != batch_dims:
            raise ValueError(
                f"incompatible shape in argument at index {i}:"
                f"expected batch shape {batch_dims} but got {arr.shape[:n_batch_dims]}"
            )

    return batch_dims


def _remove_unit_term(tensor: AlgebraT) -> AlgebraT:
    if not isinstance(tensor.basis, TensorBasis):
        raise TypeError("object must be a tensor")

    new_data = tensor.data.at[..., 0].set(0)
    return type(tensor)(new_data, tensor.basis)


def _tensor_to_dual(
    tensor: AlgebraT, new_cls: type[T], new_basis: TensorBasis | None
) -> T:
    if new_basis is None:
        new_basis = tensor.basis
    return new_cls(_redepth_data(tensor.data, new_basis.size()), new_basis)


@jax.custom_vjp
def ft_fma(a: FreeTensorT, b: FreeTensorT, c: FreeTensorT) -> FreeTensorT:
    """
    Free tensor fused multiply-add

    This function is equivalent to `b * c + a`.
    Supports float 32 or 64 but all data buffers must have matching type.

    :param a: addition operand
    :param b: left-hand multiply operand
    :param c: right-hand multiple operand
    :return: result
    """
    dtype = jnp.result_type(a.data.dtype, b.data.dtype, c.data.dtype)
    batch_dims = _get_and_check_batch_dims(a.data, b.data, c.data, core_dims=1)

    a_max_deg = a.basis.depth

    op_cls = Operation.get_operation("ft_fma", "dense")

    op = op_cls(
        (a.basis, b.basis, c.basis),
        dtype,
        batch_dims,
        a_max_deg=np.int32(a_max_deg),
        b_max_deg=np.int32(min(a_max_deg, b.basis.depth)),
        c_max_deg=np.int32(min(a_max_deg, c.basis.depth)),
        b_min_deg=np.int32(0),
        c_min_deg=np.int32(0),
    )

    out_data = op(a.data, b.data, c.data)

    return DenseFreeTensor(*out_data, op.basis)


def ft_fma_derivative(
    a: FreeTensorT,
    b: FreeTensorT,
    c: FreeTensorT,
    t_a: FreeTensorT,
    t_b: FreeTensorT,
    t_c: FreeTensorT,
) -> FreeTensorT:
    """
    Free tensor fused multiply-add derivative

    :param a: addition operand
    :param b: left-hand multiply operand
    :param c: right-hand multiple operand
    :param t_a: tangent perturbation at a
    :param t_b: tangent perturbation at b
    :param t_c: tangent perturbation at c
    :return: derivative in tangent direction (s_a, s_b, s_c)
    """
    return t_a + ft_mul_derivative(b, c, t_b, t_c)


def ft_fma_adjoint_derivative(
    a: FreeTensorT,
    b: FreeTensorT,
    c: FreeTensorT,
    ct_result: ShuffleTensorT,
) -> tuple[ShuffleTensorT, ShuffleTensorT, ShuffleTensorT]:
    """
    Free tensor fused multiply-add adjoint derivative

    :param a: addition operand
    :param b: left-hand multiply operand
    :param c: right-hand multiple operand
    :param ct_result: cotangent from the output
    :return: (cotangent for a, cotangent for b, cotangent for c)
    """
    ct_a = ct_result
    ct_b, ct_c = ft_mul_adjoint_derivative(b, c, ct_result)
    return ct_a, ct_b, ct_c


def _ft_fma_vjp_fwd(a: FreeTensorT, b: FreeTensorT, c: FreeTensorT):
    result = ft_fma(a, b, c)
    return result, (a, b, c)


def _ft_fma_vjp_bwd(residuals, ct_result_data) -> tuple[jax.Array, ...]:
    a, b, c = residuals

    if isinstance(ct_result_data, jax.Array):
        ct_result = DenseShuffleTensor(ct_result_data, a.basis)
    elif isinstance(ct_result_data, DenseShuffleTensor):
        ct_result = ct_result_data
    elif isinstance(ct_result_data, DenseFreeTensor):
        ct_result = DenseShuffleTensor(ct_result_data.data, ct_result_data.basis)
    else:
        raise TypeError(f"Unexpected type for ct_result_data: {type(ct_result_data)}")

    ct_a, ct_b, ct_c = ft_fma_adjoint_derivative(a, b, c, ct_result)
    return ct_a.data, ct_b.data, ct_c.data


ft_fma.defvjp(_ft_fma_vjp_fwd, _ft_fma_vjp_bwd)


@jax.custom_vjp
def ft_mul(a: FreeTensorT, b: FreeTensorT) -> FreeTensorT:
    """
    Free tensor multiply

    This function is equivalent to `a * b`.
    Supports float 32 or 64 but all data buffers must have matching type.

    :param a: left-hand multiply operand
    :param b: right-hand multiple operand
    :return: result
    """
    dtype = jnp.result_type(a.data.dtype, b.data.dtype)
    batch_dims = _get_and_check_batch_dims(a.data, b.data, core_dims=1)

    # Use same basis convention as ft_mul in roughpy/compute
    op_cls = Operation.get_operation("ft_mul", "dense")

    op = op_cls(
        (a.basis, b.basis),
        dtype,
        batch_dims,
        lhs_max_deg=np.int32(min(a.basis.depth, a.basis.depth)),
        rhs_max_deg=np.int32(min(a.basis.depth, b.basis.depth)),
        lhs_min_deg=np.int32(0),
        rhs_min_deg=np.int32(0),
    )

    out_data = op(a.data, b.data)

    return DenseFreeTensor(*out_data, op.basis)


def ft_mul_derivative(
    lhs: FreeTensorT, rhs: FreeTensorT, t_lhs: FreeTensorT, t_rhs: FreeTensorT
) -> FreeTensorT:
    """
    Free tensor multiply derivative (product rule).

    Dm(a,b)[s,t] = s*b + a*t  where * is the tensor product.

    :param lhs: left-hand operand
    :param rhs: right-hand operand
    :param t_lhs: tangent perturbation at lhs
    :param t_rhs: tangent perturbation at rhs
    :return: derivative in tangent direction (s,t)
    """
    return ft_mul(lhs, t_rhs) + ft_mul(t_lhs, rhs)


def ft_mul_adjoint_derivative(
    lhs: FreeTensorT, rhs: FreeTensorT, ct_result: ShuffleTensorT
) -> tuple[ShuffleTensorT, ShuffleTensorT]:
    """
    Free tensor multiply adjoint derivative.

    [Dm(a,b)*]φ = (R_b* φ, L_a* φ) where L_a and R_b are the left
    and right multiplication operators respectively.

    :param lhs: left-hand operand
    :param rhs: right-hand operand
    :param ct_result: cotangent from the output
    :return: (cotangent for lhs, cotangent for rhs)
    """
    ct_lhs = ft_adjoint_right_mul(rhs, ct_result)
    ct_rhs = ft_adjoint_left_mul(lhs, ct_result)
    return ct_lhs, ct_rhs


def _ft_mul_vjp_fwd(lhs: FreeTensorT, rhs: FreeTensorT):
    result = ft_mul(lhs, rhs)
    return result, (lhs, rhs)


def _ft_mul_vjp_bwd(residuals, ct_result_data) -> tuple[jax.Array, ...]:
    lhs, rhs = residuals

    # TODO: Not sure if this can be all different array types (a la ft_log) or
    # if it will always be DenseFreeTensor. If the latter, we can simplify this.
    if isinstance(ct_result_data, jax.Array):
        ct_result = DenseShuffleTensor(ct_result_data, lhs.basis)
    elif isinstance(ct_result_data, DenseShuffleTensor):
        ct_result = ct_result_data
    elif isinstance(ct_result_data, DenseFreeTensor):
        ct_result = DenseShuffleTensor(ct_result_data.data, ct_result_data.basis)
    else:
        raise TypeError(f"Unexpected type for ct_result_data: {type(ct_result_data)}")

    ct_lhs, ct_rhs = ft_mul_adjoint_derivative(lhs, rhs, ct_result)
    return ct_lhs.data, ct_rhs.data


ft_mul.defvjp(_ft_mul_vjp_fwd, _ft_mul_vjp_bwd)


@jax.custom_vjp
def antipode(a: AlgebraT) -> AlgebraT:
    """
    Antipode of a free tensor

    :param a: argument
    :return: new tensor with antipode of `a`
    """
    op_cls = Operation.get_operation("ft_antipode", "dense")
    batch_dims = _get_and_check_batch_dims(a.data, core_dims=1)

    out_class = type(a)

    op = op_cls(
        (a.basis,),
        a.data.dtype,
        batch_dims,
        arg_max_deg=np.int32(a.basis.depth),
        no_sign=False,
    )

    out_data = op(a.data)
    out_basis = op.basis

    return out_class(*out_data, out_basis)


def antipode_derivative(a: FreeTensorT, t_a: FreeTensorT) -> FreeTensorT:
    """
    Antipode derivative of free tensor perturbation `t_a` at `a`

    This operation is linear, with the derivative being independent of
    the argument, computed as the antipode of the tangent. This is
    because antipode is a generalisation of transpose, taking the
    equivalent of the transpose at each level, i.e. for level 1 it's
    simply flipping the sign, for level 2 it's a regular 2D transpose,
    and higher levels are similar but more complex variants of this.
    So the derivative is simply flipped through this transpose.

    :param a: argument
    :param t_a: tangent pertubation at `a`
    :return: derivative of `t_a` at `a`
    """
    return antipode(t_a)


def antipode_adjoint_derivative(
    a: FreeTensorT, ct_result: ShuffleTensorT
) -> tuple[ShuffleTensorT]:
    """
    Antipode adjoint derivative of a free tensor

    As with the antipode derivative, this is a linear operation which
    is independent of the position a, but instead operates in dual
    space to free tensor, hence shuffle tensor cotangents.

    :param a: argument
    :param ct_result: cotangent perturbation at `a`
    :return: adjoint derivative of `ct_result` at `a`
    """
    return antipode(ct_result)


def _antipode_vjp_fwd(a: FreeTensorT):
    result = antipode(a)
    return result, (a,)


def _antipode_vjp_bwd(residuals, ct_result_data: jax.Array) -> tuple[jax.Array, ...]:
    (a,) = residuals

    ct_result = DenseShuffleTensor(ct_result_data.data, ct_result_data.basis)
    ct_antipode = antipode_adjoint_derivative(a, ct_result)

    return (ct_antipode.data,)


antipode.defvjp(_antipode_vjp_fwd, _antipode_vjp_bwd)


@jax.custom_vjp
def st_fma(a: ShuffleTensorT, b: ShuffleTensorT, c: ShuffleTensorT) -> ShuffleTensorT:
    """
    Shuffle tensor fused multiply-add

    This function is equivalent to `b * c + a`.
    Supports float 32 or 64 but all data buffers must have matching type.

    :param a: input and first operand
    :param b: left-hand operand
    :param c: right-hand operand
    :return: shuffle fused multiply-add
    """
    batch_dims = _get_and_check_batch_dims(a.data, b.data, c.data, core_dims=1)
    dtype = jnp.result_type(a.data.dtype, b.data.dtype, c.data.dtype)

    op_cls = Operation.get_operation("st_fma", "dense")

    op = op_cls(
        (a.basis, b.basis, c.basis),
        dtype,
        batch_dims,
        a_max_deg=np.int32(a.basis.depth),
        b_max_deg=np.int32(min(a.basis.depth, b.basis.depth)),
        c_max_deg=np.int32(min(b.basis.depth, c.basis.depth)),
        b_min_deg=np.int32(0),
        c_min_deg=np.int32(0),
    )
    out_data = op(a.data, b.data, c.data)

    return DenseShuffleTensor(*out_data, op.basis)


def st_fma_derivative(
    a: ShuffleTensorT,
    b: ShuffleTensorT,
    c: ShuffleTensorT,
    t_a: ShuffleTensorT,
    t_b: ShuffleTensorT,
    t_c: ShuffleTensorT,
) -> ShuffleTensorT:
    check_basis_compat(a.basis, b.basis, c.basis, t_a.basis, t_b.basis, t_c.basis)
    _get_and_check_batch_dims(
        a.data, b.data, c.data, t_a.data, t_b.data, t_c.data, core_dims=1
    )

    return t_a + st_mul_derivative(b, c, t_b, t_c)


def st_fma_adjoint_derivative(
    a: ShuffleTensorT,
    b: ShuffleTensorT,
    c: ShuffleTensorT,
    ct_result: FreeTensorT,
) -> tuple[FreeTensorT, FreeTensorT, FreeTensorT]:
    check_basis_compat(a.basis, b.basis, c.basis, ct_result.basis)
    _get_and_check_batch_dims(a.data, b.data, c.data, ct_result.data, core_dims=1)

    ct_a = ct_result  # TODO: Map explicitly to adjoint type
    ct_b, ct_c = st_mul_adjoint_derivative(b, c, ct_result)

    return ct_a, ct_b, ct_c


def _st_fma_vjp_fwd(
    a: ShuffleTensorT, b: ShuffleTensorT, c: ShuffleTensorT
) -> tuple[ShuffleTensorT, tuple[ShuffleTensorT, ShuffleTensorT, ShuffleTensorT]]:
    result = st_fma(a, b, c)
    return result, (a, b, c)


def _st_fma_vjp_bwd(
    residuals: tuple[ShuffleTensorT, ShuffleTensorT, ShuffleTensorT],
    ct_result_data: jax.Array | DenseFreeTensor | DenseShuffleTensor,
) -> tuple[jax.Array, ...]:
    a, b, c = residuals

    if isinstance(ct_result_data, jax.Array):
        ct_result = DenseFreeTensor(ct_result_data, a.basis)
    elif isinstance(ct_result_data, DenseFreeTensor):
        ct_result = ct_result_data
    elif isinstance(ct_result_data, DenseShuffleTensor):
        ct_result = DenseFreeTensor(ct_result_data.data, ct_result_data.basis)
    else:
        raise TypeError(f"Unexpected type for ct_result_data: {type(ct_result_data)}")

    ct_a, ct_b, ct_c = st_fma_adjoint_derivative(a, b, c, ct_result)

    return ct_a.data, ct_b.data, ct_c.data


st_fma.defvjp(_st_fma_vjp_fwd, _st_fma_vjp_bwd)


@jax.custom_vjp
def st_mul(lhs: ShuffleTensorT, rhs: ShuffleTensorT) -> ShuffleTensorT:
    """
    Shuffle tensor product

    This function is equivalent to `lhs & rhs`.
    Supports float 32 or 64 but all data buffers must have matching type.

    :param lhs: left-hand operand
    :param rhs: right-hand operand
    :return: the shuffle product of lhs and rhs
    """
    dtype = jnp.result_type(lhs.data.dtype, rhs.data.dtype)
    batch_dims = _get_and_check_batch_dims(lhs.data, rhs.data, core_dims=1)

    op_cls = Operation.get_operation("st_mul", "dense")

    op = op_cls(
        (lhs.basis, rhs.basis),
        dtype,
        batch_dims,
        lhs_max_deg=np.int32(lhs.basis.depth),
        rhs_max_deg=np.int32(rhs.basis.depth),
        lhs_min_deg=np.int32(0),
        rhs_min_deg=np.int32(0),
    )

    out_data = op(lhs.data, rhs.data)

    return DenseShuffleTensor(*out_data, op.basis)


def st_mul_derivative(
    lhs: ShuffleTensorT,
    rhs: ShuffleTensorT,
    t_lhs: ShuffleTensorT,
    t_rhs: ShuffleTensorT,
) -> ShuffleTensorT:
    check_basis_compat(lhs.basis, rhs.basis, t_lhs.basis, t_rhs.basis)
    _get_and_check_batch_dims(lhs.data, rhs.data, t_lhs.data, t_rhs.data)

    t_result = st_mul(lhs, t_rhs) + st_mul(t_lhs, rhs)
    return t_result


def st_mul_adjoint_derivative(
    lhs: ShuffleTensorT, rhs: ShuffleTensorT, ct_result: FreeTensorT
) -> tuple[FreeTensorT, FreeTensorT]:
    check_basis_compat(lhs.basis, rhs.basis, ct_result.basis)
    _get_and_check_batch_dims(lhs.data, rhs.data, ct_result.data)

    ct_lhs = st_adjoint_mul(rhs, ct_result)
    ct_rhs = st_adjoint_mul(lhs, ct_result)

    return ct_lhs, ct_rhs


def _st_mul_vjp_fwd(lhs, rhs):
    result = st_mul(lhs, rhs)
    return result, (lhs, rhs)


def _st_mul_vjp_bwd(residuals, ct_result):
    lhs, rhs = residuals
    ct_lhs, ct_rhs = st_mul_adjoint_derivative(lhs, rhs, ct_result)
    return ct_lhs.data, ct_rhs.data


st_mul.defvjp(_st_mul_vjp_fwd, _st_mul_vjp_bwd)


# @partial(jax.custom_vjp, nondiff_argnums=(1,))
@jax.custom_vjp
def ft_exp(x: FreeTensorT, out_basis: TensorBasis | None = None) -> FreeTensorT:
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
    dtype = x.data.dtype

    op_cls = Operation.get_operation("ft_exp", "dense")
    op = op_cls(
        (x.basis,),
        dtype,
        x.batch_shape,
        specific_basis=out_basis,
        arg_max_deg=np.int32(x.basis.depth),
    )

    out_data = op(x.data)
    out_basis = op.basis

    return DenseFreeTensor(*out_data, out_basis)


def ft_exp_derivative(
    x: FreeTensorT,
    t_x: FreeTensorT,
    out_basis: TensorBasis | None = None,
) -> FreeTensorT:
    check_basis_compat(x.basis, t_x.basis)
    batch_dims = _get_and_check_batch_dims(x.data, t_x.data, core_dims=1)
    dtype = jnp.result_type(x.data.dtype, t_x.data.dtype)

    x = _remove_unit_term(x)

    # The ft_exp function is really the composition of exp with the projection onto the non-unit
    # terms, so we must apply this projection to the incoming tangent as well.
    t_x = _remove_unit_term(t_x)

    basis = out_basis or x.basis
    depth = basis.depth

    r_d_data = jnp.zeros((*batch_dims, basis.size()), dtype=dtype).at[..., 0].set(1)
    r_d = DenseFreeTensor(r_d_data, basis)
    t_r_d_data = jnp.zeros((*batch_dims, basis.size()), dtype=dtype)
    t_r_d = DenseFreeTensor(t_r_d_data, basis)

    for d in range(depth, 0, -1):
        scale = 1.0 / d
        r_dm1 = scale * ft_mul(x, r_d)
        r_dm1.data = r_dm1.data.at[..., 0].add(1)

        t_r_dm1 = scale * (ft_mul(t_x, r_d) + ft_mul(x, t_r_d))

        r_d = r_dm1
        t_r_d = t_r_dm1

    return t_r_d


def ft_exp_adjoint_derivative(
    x: FreeTensorT,
    ct_result: ShuffleTensorT,
    out_basis: TensorBasis | None = None,
) -> tuple[ShuffleTensorT]:
    check_basis_compat(x.basis, ct_result.basis)
    batch_dims = _get_and_check_batch_dims(x.data, ct_result.data, core_dims=1)
    dtype = jnp.result_type(x.data.dtype, ct_result.data.dtype)

    x = _remove_unit_term(x)

    basis = out_basis or ct_result.basis
    depth = basis.depth

    ident_data = jnp.zeros((*batch_dims, basis.size()), dtype=dtype).at[..., 0].set(1)
    ident = ShuffleTensor(ident_data, basis)

    r_data = [None for _ in range(depth)] + [ident]
    for d in range(depth, 0, -1):
        scale = 1.0 / d
        # noinspection PyTypeChecker
        r_data[d - 1] = scale * ft_mul(x, r_data[d])
        r_data[d - 1].data = r_data[d - 1].data.at[..., 0].add(1)

    ct_x_data = jnp.zeros((*batch_dims, basis.size()), dtype=dtype)
    ct_x = DenseShuffleTensor(ct_x_data, basis)
    ct_r = ct_result

    for d in range(1, depth + 1):
        scale = 1.0 / d
        ct_x = ct_x + scale * ft_adjoint_left_mul(r_data[d], ct_r)
        ct_r = scale * ft_adjoint_right_mul(x, ct_r)

    # The function ft_exp is actually exp composed with the projection onto the non-unit terms
    # so the adjoint derivative needs to apply this to the resulting cotangent.
    ct_x = _remove_unit_term(ct_x)

    return (ct_x,)


def _ft_exp_vjp_fwd(
    x: FreeTensorT, out_basis: TensorBasis | None = None
) -> FreeTensorT:

    result = ft_exp(x, out_basis=out_basis)
    return result, (x, result)


def _ft_exp_vjp_bwd(
    residuals: tuple[Any, ...],
    ct_result_data: ShuffleTensorT,
) -> tuple[jax.Array | None, ...]:
    x, result = residuals

    if isinstance(ct_result_data, DenseShuffleTensor):
        ct_result = ct_result_data
    elif isinstance(ct_result_data, DenseFreeTensor):
        ct_result = DenseShuffleTensor(ct_result_data.data, ct_result_data.basis)
    elif isinstance(ct_result_data, jax.Array):
        ct_result = DenseShuffleTensor(ct_result_data, result.basis)
    else:
        raise TypeError(f"unexpected type for ct_result_data: {type(ct_result_data)}")

    (ct_x,) = ft_exp_adjoint_derivative(x, ct_result)
    return ct_x.data, None


ft_exp.defvjp(_ft_exp_vjp_fwd, _ft_exp_vjp_bwd)


@jax.custom_vjp
def ft_log(x: FreeTensorT, out_basis: TensorBasis | None = None) -> FreeTensorT:
    """
    Free tensor logarithm

    This function computes `log(1 + x)` through the (truncated) power series
    expansion in `x`. This function assumes that `x` is actually of the form `1 + x`, and
    the power series computation simply ignores the unit coefficient if it is present. This
    means that the result of `ft_log(x)` where `x` has a unit term different from 1 might
    be different than expected. The intention is that this function should only be applied
    to tensors that are known to be of the form `1 + x`, such as group-like elements such as
    those computed using `ft_exp`.

    Supports float 32 or 64 but all data buffers must have matching type.
    If `out_basis` is not specified, the same basis as `x` is used.

    :param x: argument
    :param out_basis: optional output basis.
    :return: tensor logarithm of `x`
    """
    _check_tensor_dtype(x)
    dtype = x.data.dtype

    op_cls = Operation.get_operation("ft_log", "dense")
    op = op_cls(
        (x.basis,),
        dtype,
        x.batch_shape,
        specific_basis=out_basis,
        arg_max_deg=np.int32(x.basis.depth),
    )

    out_data = op(x.data)
    out_basis = op.basis

    return DenseFreeTensor(*out_data, out_basis)


def ft_log_derivative(
    x: FreeTensorT,
    t_x: FreeTensorT,
) -> FreeTensorT:
    check_basis_compat(x.basis, t_x.basis)
    batch_dims = _get_and_check_batch_dims(x.data, t_x.data, core_dims=1)
    dtype = jnp.result_type(x.data.dtype, t_x.data.dtype)

    x = _remove_unit_term(x)
    t_x = _remove_unit_term(t_x)

    basis = x.basis
    depth = basis.depth

    r_d_data = jnp.zeros((*batch_dims, basis.size()), dtype=dtype)
    r_d = DenseFreeTensor(r_d_data, basis)
    t_r_d_data = jnp.zeros((*batch_dims, basis.size()), dtype=dtype)
    t_r_d = DenseFreeTensor(t_r_d_data, basis)

    for d in range(depth, 0, -1):
        sign = -1 if d % 2 == 0 else 1
        r_d.data = r_d.data.at[..., 0].add(sign / d)

        r_dm1 = ft_mul(x, r_d)
        t_r_d = ft_mul(t_x, r_d) + ft_mul(x, t_r_d)
        r_d = r_dm1

    return t_r_d


def ft_log_adjoint_derivative(
    x: FreeTensorT,
    ct_result: ShuffleTensorT,
) -> tuple[ShuffleTensorT]:
    check_basis_compat(x.basis, ct_result.basis)
    batch_dims = _get_and_check_batch_dims(x.data, ct_result.data, core_dims=1)
    dtype = jnp.result_type(x.data.dtype, ct_result.data.dtype)

    x = _remove_unit_term(x)

    basis = x.basis
    depth = basis.depth

    zero_data = jnp.zeros((*batch_dims, basis.size()), dtype=dtype)
    zero = DenseFreeTensor(zero_data, basis)
    rs = [None for _ in range(depth)] + [zero]
    for d in range(depth, 0, -1):
        sign = -1 if d % 2 == 0 else 1
        r_data = rs[d].data.at[..., 0].add(sign / d)
        r = DenseFreeTensor(r_data, basis)
        rs[d - 1] = ft_mul(x, r)

    ct_x_data = jnp.zeros((*batch_dims, basis.size()), dtype=dtype)
    ct_x = DenseShuffleTensor(ct_x_data, basis)
    ct_r_d = ct_result

    for d in range(1, depth + 1):
        sign = -1 if d % 2 == 0 else 1
        u_d_data = rs[d].data.at[..., 0].add(sign / d)
        u_d = DenseFreeTensor(u_d_data, basis)

        ct_x = ct_x + ft_adjoint_right_mul(u_d, ct_r_d)
        ct_r_d = ft_adjoint_left_mul(x, ct_r_d)

    return (ct_x,)


def _ft_log_vjp_fwd(
    x: FreeTensorT, out_basis: TensorBasis | None = None
) -> tuple[FreeTensorT, tuple[Any, ...]]:
    result = ft_log(x, out_basis=out_basis)
    return result, (x, result)


def _ft_log_vjp_bwd(
    residuals: tuple[Any, ...], ct_result_data: Any
) -> tuple[jax.Array | None, ...]:
    x, result = residuals

    if isinstance(ct_result_data, jax.Array):
        ct_result = DenseShuffleTensor(ct_result_data, result.basis)
    elif isinstance(ct_result_data, DenseShuffleTensor):
        ct_result = ct_result_data
    elif isinstance(ct_result_data, DenseFreeTensor):
        ct_result = DenseShuffleTensor(ct_result_data.data, ct_result_data.basis)
    else:
        raise TypeError(f"Unexpected type for ct_result_data: {type(ct_result_data)}")

    (ct_x,) = ft_log_adjoint_derivative(x, ct_result)
    return (ct_x.data, None)


ft_log.defvjp(_ft_log_vjp_fwd, _ft_log_vjp_bwd)


@jax.custom_vjp
def ft_fmexp(
    multiplier: FreeTensorT, exponent: FreeTensorT, out_basis: TensorBasis | None = None
) -> FreeTensorT:
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
    dtype = multiplier.data.dtype

    batch_dims = _get_and_check_batch_dims(multiplier.data, exponent.data, core_dims=1)

    mul_depth = multiplier.basis.depth
    exp_depth = exponent.basis.depth

    op_cls = Operation.get_operation("ft_fmexp", "dense")
    op = op_cls(
        (multiplier.basis, exponent.basis),
        dtype,
        batch_dims,
        specific_basis=out_basis,
        mul_max_deg=np.int32(mul_depth),
        exp_max_deg=np.int32(exp_depth),
        mul_min_deg=np.int32(0),
        exp_min_deg=np.int32(0),
    )

    out_data = op(multiplier.data, exponent.data)
    out_basis = op.basis

    return DenseFreeTensor(*out_data, out_basis)


def ft_fmexp_derivative(
    multiplier: FreeTensorT,
    exponent: FreeTensorT,
    t_multiplier: FreeTensorT,
    t_exponent: FreeTensorT,
) -> FreeTensorT:
    check_basis_compat(
        multiplier.basis, exponent.basis, t_multiplier.basis, t_exponent.basis
    )
    _get_and_check_batch_dims(
        multiplier.data, exponent.data, t_multiplier.data, t_exponent.data, core_dims=1
    )

    exponent = _remove_unit_term(exponent)
    t_exponent = _remove_unit_term(t_exponent)

    basis = multiplier.basis
    depth = basis.depth

    r_d = multiplier
    t_r_d = t_multiplier

    for d in range(depth, 0, -1):
        scale = 1.0 / d
        r_dm1 = multiplier + scale * ft_mul(r_d, exponent)
        t_r_dm1 = t_multiplier + scale * (
            ft_mul(r_d, t_exponent) + ft_mul(t_r_d, exponent)
        )

        r_d = r_dm1
        t_r_d = t_r_dm1

    return t_r_d


def ft_fmexp_adjoint_derivative(
    multiplier: FreeTensorT,
    exponent: FreeTensorT,
    ct_result: ShuffleTensorT,
) -> tuple[ShuffleTensorT, ShuffleTensorT]:
    check_basis_compat(multiplier.basis, exponent.basis, ct_result.basis)
    _get_and_check_batch_dims(
        multiplier.data, exponent.data, ct_result.data, core_dims=1
    )

    # tensor_type = type(multiplier)
    ct_type = type(ct_result)

    basis = multiplier.basis
    depth = basis.depth

    exponent = _remove_unit_term(exponent)

    # First recompute the intermediate approximations of the fmexp. The computation
    # of the cotangents proceeds in the opposite order, from 0 to D instead of from
    # D to 0. These are fairly cheap to recompute in theory because of the higher
    # terms "dropping off the cliff" as the degree increases. This optimisation is
    # not implemented here yet, because we haven't built the support into the high
    # level functions of disregarding higher-order terms yet.

    r_data = [None for _ in range(depth)] + [multiplier]
    for d in range(depth, 0, -1):
        scale = 1.0 / d
        # noinspection PyTypeChecker
        r_data[d - 1] = multiplier + scale * ft_mul(exponent, r_data[d])

    # Now we can update the cotangents in sequence. We have to accumulate the
    # ct_multiplier and ct_exponent values, as well as the cotangents of
    # each of the intermediate r terms.

    ct_multiplier = ct_type(jnp.zeros_like(multiplier.data), multiplier.basis)
    ct_exponent = ct_type(jnp.zeros_like(exponent.data), exponent.basis)
    ct_r = ct_result

    for d in range(1, depth + 1):
        scale = 1.0 / d
        ct_multiplier = ct_multiplier + ct_r
        ct_exponent = ct_exponent + scale * ft_adjoint_left_mul(r_data[d], ct_r)
        ct_r = scale * ft_adjoint_right_mul(exponent, ct_r)

    ct_multiplier = ct_multiplier + ct_r

    ct_exponent = _remove_unit_term(ct_exponent)

    return ct_multiplier, ct_exponent


def _ft_fmexp_vjp_fwd(
    multiplier: FreeTensorT, exponent: FreeTensorT, out_basis: TensorBasis | None
):
    result = ft_fmexp(multiplier, exponent, out_basis)
    return result, (multiplier, exponent, result)


def _ft_fmexp_vjp_bwd(
    residuals, ct_result_data: jax.Array
) -> tuple[jax.Array | None, ...]:
    multiplier, exponent, result = residuals
    if isinstance(ct_result_data, jax.Array):
        ct_result = DenseShuffleTensor(ct_result_data, result.basis)
    elif isinstance(ct_result_data, DenseShuffleTensor):
        ct_result = ct_result_data
    elif isinstance(ct_result_data, DenseFreeTensor):
        ct_result = DenseShuffleTensor(ct_result_data.data, ct_result_data.basis)
    else:
        raise TypeError(f"Unexpected type for ct_result_data: {type(ct_result_data)}")

    ct_multiplier, ct_exponent = ft_fmexp_adjoint_derivative(
        multiplier, exponent, ct_result
    )
    return ct_multiplier.data, ct_exponent.data, None


ft_fmexp.defvjp(_ft_fmexp_vjp_fwd, _ft_fmexp_vjp_bwd)


@jax.custom_vjp
def lie_to_tensor(arg: LieT, scale_factor=None) -> FreeTensorT:
    """
    Compute the embedding of a Lie algebra element as a free tensor.

    :param arg: Lie to embed into the tensor algebra.
    :param scale_factor: Optional scalar multiplier applied to the embedded tensor.
    :return: New FreeTensor containing the embedding of ``arg``.
    """
    if not isinstance(arg, Lie):
        raise ValueError(f"Invalid lie_to_tensor arg type {type(arg)}")

    _check_tensor_dtype(arg)
    dtype = arg.data.dtype

    op_cls = Operation.get_operation("lie_to_tensor", "dense")
    op = op_cls(
        (arg.basis,),
        dtype,
        arg.batch_shape,
        scale_factor=scale_factor,
    )

    out_data = op(arg.data)
    out_basis = op.basis

    return DenseFreeTensor(*out_data, out_basis)


def lie_to_tensor_derivative(
    arg: LieT,
    t_arg: LieT,
    scale_factor=None,
) -> FreeTensorT:
    """
    Lie to tensor derivative of free tensor perturbation `t_arg` at `arg`
    """
    return lie_to_tensor(t_arg, scale_factor)


def lie_to_tensor_adjoint_derivative(
    arg: LieT,
    ct_result: ShuffleTensorT,
    scale_factor=None,
) -> tuple[LieT]:
    """
    Lie to tensor derivative of free tensor `ct_result` at `arg`
    """
    l2t = arg.basis.get_l2t_matrix(arg.data.dtype)
    l2t_size = arg.basis.size()
    data = csr_matvec(l2t.data, l2t.indices, l2t.indptr, l2t_size, ct_result.data)
    if scale_factor:
        data = data * scale_factor

    return DenseLie(data, arg.basis)


def _lie_to_tensor_vjp_fwd(arg: LieT, scale_factor=None):
    result = lie_to_tensor_derivative(arg, scale_factor)
    return result, (arg, scale_factor)


def _lie_to_tensor_vjp_bwd(
    residuals, ct_result_data: jax.Array
) -> tuple[jax.Array, ...]:
    arg, scale_factor = residuals

    ct_result = Lie(ct_result_data.data, ct_result_data.basis)
    ct_l2t_adjoint_deriv = lie_to_tensor_adjoint_derivative(
        arg, ct_result, scale_factor
    )

    return (ct_l2t_adjoint_deriv.data,)


lie_to_tensor.defvjp(_lie_to_tensor_vjp_fwd, _lie_to_tensor_vjp_bwd)


@jax.custom_vjp
def tensor_to_lie(arg: FreeTensorT, scale_factor=None) -> LieT:
    """
    Project a free tensor onto the embedding of the Lie algebra in the tensor algebra.

    :param arg: Free tensor to project into the Lie algebra.
    :param scale_factor: Optional scalar multiplier applied to the projected Lie element.
    :return: New Lie containing the projection of ``arg``.
    """
    if not isinstance(arg, FreeTensor):
        raise ValueError(f"Invalid lie_to_tensor arg type {type(arg)}")

    _check_tensor_dtype(arg)
    dtype = arg.data.dtype

    op_cls = Operation.get_operation("tensor_to_lie", "dense")
    op = op_cls(
        (arg.basis,),
        dtype,
        arg.batch_shape,
        scale_factor=scale_factor,
    )

    out_data = op(arg.data)
    out_basis = op.basis

    return DenseLie(*out_data, out_basis)


def tensor_to_lie_derivative(
    arg: FreeTensorT,
    t_arg: FreeTensorT,
    scale_factor=None,
) -> LieT:
    """
    Tensor to Lie derivative of Lie perturbation `t_arg` at `arg`

    Since tensor_to_lie is a linear map T2L, its derivative is
    independent of the position and is simply T2L applied to the
    tangent direction.
    """
    return tensor_to_lie(t_arg, scale_factor)


def tensor_to_lie_adjoint_derivative(
    arg: FreeTensorT,
    ct_result: LieT,
    scale_factor=None,
) -> tuple[ShuffleTensorT]:
    """
    Tensor to Lie adjoint derivative of Lie cotangent `ct_result` at `arg`

    Computes T2L^T applied to the cotangent. The transpose is obtained
    by feeding the CSC-stored t2l matrix data into csr_matvec, which
    implicitly transposes the matrix.
    """
    # TODO: consider changing basis resolution logic
    lie_basis = to_lie_basis(arg.basis)
    t2l = lie_basis.get_t2l_matrix(arg.data.dtype)
    t2l_size = arg.basis.size()
    data = csr_matvec(t2l.data, t2l.indices, t2l.indptr, t2l_size, ct_result.data)
    if scale_factor:
        data = data * scale_factor

    return DenseShuffleTensor(data, arg.basis)


def _tensor_to_lie_vjp_fwd(arg: FreeTensorT, scale_factor=None):
    result = tensor_to_lie(arg, scale_factor)
    return result, (arg, scale_factor)


def _tensor_to_lie_vjp_bwd(
    residuals, ct_result_data: jax.Array
) -> tuple[jax.Array, ...]:
    arg, scale_factor = residuals

    if isinstance(ct_result_data, DenseLie):
        ct_result = ct_result_data
    elif isinstance(ct_result_data, DenseFreeTensor):
        ct_result = DenseLie(ct_result_data.data, ct_result_data.basis)
    else:
        ct_result = DenseLie(ct_result_data.data, ct_result_data.basis)

    ct_t2l_adjoint_deriv = tensor_to_lie_adjoint_derivative(
        arg, ct_result, scale_factor
    )

    return (ct_t2l_adjoint_deriv.data, None)


tensor_to_lie.defvjp(_tensor_to_lie_vjp_fwd, _tensor_to_lie_vjp_bwd)


@jax.custom_vjp
def ft_adjoint_left_mul(op: FreeTensorT, arg: ShuffleTensorT) -> ShuffleTensorT:
    """
    Compute the adjoint action of left free-tensor multiplication on shuffles.

    Computes the adjoint action of the left multiplier operator L_a: b -> ab
    as an operator on shuffle tensors. That is, the shuffle tensor given
    by L_a^*(s) where * denotes adjoint.

    :param a: a in the notation above
    :param arg: The ShuffleTensor to be acted upon.
    :return: The result of the adjoint action as a ShuffleTensor.
    """
    dtype = jnp.result_type(op.data.dtype, arg.data.dtype)
    batch_dims = _get_and_check_batch_dims(op.data, arg.data, core_dims=1)

    op_max_deg = op.basis.depth
    arg_max_deg = arg.basis.depth

    op_cls = Operation.get_operation("ft_adj_lmul", "dense")
    op_call = op_cls(
        (op.basis,),
        dtype,
        batch_dims,
        op_max_deg=np.int32(op_max_deg),
        arg_max_deg=np.int32(arg_max_deg),
    )

    out_data = op_call(op.data, arg.data)
    out_basis = op.basis

    return DenseShuffleTensor(*out_data, out_basis)


def ft_adjoint_left_mul_derivative(
    op: FreeTensorT,
    arg: ShuffleTensorT,
    t_op: FreeTensorT,
    t_arg: ShuffleTensorT,
) -> ShuffleTensorT:
    check_basis_compat(op.basis, arg.basis, t_op.basis, t_arg.basis)
    _get_and_check_batch_dims(op.data, arg.data, t_op.data, t_arg.data, core_dims=1)

    t_result = ft_adjoint_left_mul(t_op, arg) + ft_adjoint_left_mul(op, t_arg)

    return t_result


def ft_adjoint_left_mul_adjoint_derivative(
    op: FreeTensorT, arg: ShuffleTensorT, ct_result: FreeTensorT
) -> tuple[ShuffleTensorT, FreeTensorT]:
    check_basis_compat(op.basis, arg.basis, ct_result.basis)
    _get_and_check_batch_dims(op.data, arg.data, ct_result.data, core_dims=1)

    ct_op = ft_adjoint_right_mul(ct_result, arg)
    ct_arg = ft_mul(op, ct_result)

    return ct_op, ct_arg


def _ft_adjoint_left_mul_vjp_fwd(op: FreeTensorT, arg: ShuffleTensorT):
    ct_result = ft_adjoint_left_mul_adjoint_derivative(op, arg)
    return ct_result, (op, arg)


def _ft_adjoint_left_mul_vjp_bwd(residuals, ct_result):
    op, arg = residuals

    ct_op, ct_arg = ft_adjoint_left_mul_adjoint_derivative(op, arg, ct_result)
    return ct_op.data, ct_arg.data


ft_adjoint_left_mul.defvjp(_ft_adjoint_left_mul_vjp_fwd, _ft_adjoint_left_mul_vjp_bwd)


@jax.custom_vjp
def ft_adjoint_right_mul(op: FreeTensorT, arg: ShuffleTensorT) -> ShuffleTensorT:
    """
    Compute the adjoint action of right free-tensor multiplication on shuffles.

    Computes the adjoint action of the right multiplier operator R_a: b -> ba
    as an operator on shuffle tensors. That is, the shuffle tensor given
    by R_a^*(s) where * denotes adjoint.

    :param op: The FreeTensor representing the Lie algebra element.
    :param arg: The ShuffleTensor to be acted upon.
    :return: The result of the adjoint action as a ShuffleTensor.
    """
    dtype = jnp.result_type(op.data.dtype, arg.data.dtype)

    batch_dims = _get_and_check_batch_dims(op.data, arg.data, core_dims=1)

    op_max_deg = op.basis.depth
    arg_max_deg = arg.basis.depth

    op_cls = Operation.get_operation("ft_adj_rmul", "dense")
    op_call = op_cls(
        (op.basis,),
        dtype,
        batch_dims,
        op_max_deg=np.int32(op_max_deg),
        arg_max_deg=np.int32(arg_max_deg),
    )

    out_data = op_call(op.data, arg.data)
    out_basis = op.basis

    return DenseShuffleTensor(*out_data, out_basis)


def ft_adjoint_right_mul_derivative(
    op: FreeTensorT,
    arg: ShuffleTensorT,
    t_op: FreeTensorT,
    t_arg: ShuffleTensorT,
) -> ShuffleTensorT:
    check_basis_compat(op.basis, arg.basis, t_op.basis, t_arg.basis)
    _get_and_check_batch_dims(op.data, arg.data, core_dims=1)

    t_result = ft_adjoint_right_mul(t_op, arg) + ft_adjoint_right_mul(op, t_arg)
    return t_result


def ft_adjoint_right_mul_adjoint_derivative(
    op: FreeTensorT, arg: ShuffleTensorT, ct_result: FreeTensorT
) -> tuple[ShuffleTensorT, FreeTensorT]:
    check_basis_compat(op.basis, arg.basis, ct_result.basis)
    _get_and_check_batch_dims(op.data, arg.data, ct_result.data, core_dims=1)

    ct_op = ft_adjoint_left_mul(ct_result, arg)
    ct_arg = ft_mul(ct_result, op)

    return ct_op, ct_arg


def _ft_adjoint_right_mul_vjp_fwd(op, arg):
    result = ft_adjoint_right_mul(op, arg)
    return result, (op, arg)


def _ft_adjoint_right_mul_vjp_bwd(residuals, ct_result):
    op, arg = residuals
    ct_op, ct_arg = ft_adjoint_right_mul_adjoint_derivative(op, arg, ct_result)
    return ct_op.data, ct_arg.data


ft_adjoint_right_mul.defvjp(
    _ft_adjoint_right_mul_vjp_fwd, _ft_adjoint_right_mul_vjp_bwd
)


@jax.custom_vjp
def tensor_pairing(functional: ShuffleTensorT, argument: FreeTensorT) -> jax.Array:
    """
    Computes the tensor pairing between a functional tensor and a free tensor.

    The pairing is the evaluation of a functional (shuffle tensor) on a free tensor
    argument. The result of such a pairing is a scalar. However, to accommodate for
    internal batching, the result is returned as a JAX Array.

    Both input tensors must be compatible in terms of their basis and batch dimensions.
    The function returns a JAX Array whose shape is the same as the batch dimensions.

    :param functional: A `ShuffleTensorT` object representing the functional tensor.
        Its data will be used in the tensor pairing operation.
    :param argument: A `FreeTensorT` object representing the free tensor to be
        paired with the functional tensor.
    :return: A `jax.Array` containing the result of the tensor pairing operation.
    """
    dtype = jnp.result_type(functional.data.dtype, argument.data.dtype)
    batch_dims = _get_and_check_batch_dims(functional.data, argument.data, core_dims=1)

    op_cls = Operation.get_operation("tensor_pairing", "dense")

    op = op_cls(
        (functional.basis, argument.basis),
        dtype,
        batch_dims,
        functional_max_degree=functional.basis.depth,
        argument_max_degree=argument.basis.depth,
    )

    (result,) = op(functional.data, argument.data)

    return result


def tensor_pairing_derivative(
    functional: ShuffleTensorT,
    argument: FreeTensorT,
    t_functional: ShuffleTensorT,
    t_argument: FreeTensorT,
) -> jax.Array:
    check_basis_compat(
        functional.basis, argument.basis, t_functional.basis, t_argument.basis
    )
    _ = _get_and_check_batch_dims(
        functional.data, argument.data, t_functional.data, t_argument.data, core_dims=1
    )
    x = tensor_pairing(functional, t_argument)
    y = tensor_pairing(t_functional, argument)
    return x + y


def _reshape_pairing_cotangent(
    ct_result: jax.Array, batch_dims: tuple[int, ...]
) -> jax.Array:
    ct_shape = ct_result.shape
    if len(ct_shape) > len(batch_dims) or ct_shape != batch_dims[: len(ct_shape)]:
        raise ValueError(
            f"incompatible shapes: {ct_shape} and {batch_dims[: len(ct_shape)]}"
        )

    new_shape = ct_shape + (1,) * (len(batch_dims) - len(ct_shape)) + (1,)
    return jnp.reshape(ct_result, new_shape)


def tensor_pairing_adjoint_derivative(
    functional: ShuffleTensorT,
    argument: FreeTensorT,
    ct_result: jax.Array,
) -> tuple[FreeTensorT, ShuffleTensorT]:
    check_basis_compat(functional.basis, argument.basis)
    batch_dims = _get_and_check_batch_dims(functional.data, argument.data, core_dims=1)

    ext_ct = _reshape_pairing_cotangent(ct_result, batch_dims)

    ct_functional = DenseFreeTensor(ext_ct * argument.data, functional.basis)
    ct_argument = DenseShuffleTensor(ext_ct * functional.data, argument.basis)

    return ct_functional, ct_argument


def _tensor_pairing_vjp_fwd(functional, argument):
    result = tensor_pairing(functional, argument)
    return result, (functional, argument)


def _tensor_pairing_vjp_bwd(residual, ct_result):
    functional, argument = residual
    ct_functional, ct_argument = tensor_pairing_adjoint_derivative(
        functional, argument, ct_result
    )

    return ct_functional.data, ct_argument.data


tensor_pairing.defvjp(_tensor_pairing_vjp_fwd, _tensor_pairing_vjp_bwd)


@jax.custom_vjp
def lie_pairing(functional: LieT, argument: LieT) -> jax.Array:
    """
    Compute the pairing of two Lie algebra elements.

    This is the coefficient-space pairing induced by the Hall basis. The result
    is scalar-valued, with any leading batch dimensions preserved.

    :param functional: Left-hand Lie element.
    :param argument: Right-hand Lie element.
    :return: Pairing of ``functional`` and ``argument``.
    """
    dtype = jnp.result_type(functional.data.dtype, argument.data.dtype)
    batch_dims = _get_and_check_batch_dims(functional.data, argument.data, core_dims=1)

    op_cls = Operation.get_operation("lie_pairing", "dense")

    op = op_cls(
        (functional.basis, argument.basis),
        dtype,
        batch_dims,
        functional_max_degree=functional.basis.depth,
        argument_max_degree=argument.basis.depth,
    )

    (result,) = op(functional.data, argument.data)
    return result


def lie_pairing_derivative(
    functional: LieT,
    argument: LieT,
    t_functional: LieT,
    t_argument: LieT,
) -> jax.Array:
    check_basis_compat(
        functional.basis, argument.basis, t_functional.basis, t_argument.basis
    )
    _ = _get_and_check_batch_dims(
        functional.data, argument.data, t_functional.data, t_argument.data, core_dims=1
    )

    x = lie_pairing(functional, t_argument)
    y = lie_pairing(t_functional, argument)
    return x + y


def lie_pairing_adjoint_derivative(
    functional: LieT, argument: LieT, ct_result: jax.Array
) -> tuple[LieT, LieT]:
    check_basis_compat(functional.basis, argument.basis)
    batch_dims = _get_and_check_batch_dims(functional.data, argument.data, core_dims=1)

    ext_ct = _reshape_pairing_cotangent(ct_result, batch_dims)

    ct_functional = DenseLie(ext_ct * argument.data, functional.basis)
    ct_argument = DenseLie(ext_ct * functional.data, argument.basis)

    return ct_functional, ct_argument


def _lie_pairing_vjp_fwd(functional: LieT, argument: LieT) -> tuple[jax.Array, Any]:
    result = lie_pairing(functional, argument)
    return result, (functional, argument)


def _lie_pairing_vjp_bwd(residuals, ct_result) -> tuple[jax.Array, ...]:
    functional, argument = residuals

    ct_functional, ct_argument = lie_pairing_adjoint_derivative(
        functional, argument, ct_result
    )
    return ct_functional.data, ct_argument.data


lie_pairing.defvjp(_lie_pairing_vjp_fwd, _lie_pairing_vjp_bwd)


@jax.custom_vjp
def st_adjoint_mul(
    op_arg: ShuffleTensorT,
    arg: FreeTensorT,
) -> FreeTensorT:
    dtype = jnp.result_type(op_arg.data.dtype, arg.data.dtype)
    check_basis_compat(op_arg.basis, arg.basis)
    batch_dims = _get_and_check_batch_dims(op_arg.data, arg.data, core_dims=1)

    op_cls = Operation.get_operation("st_adj_mul", "dense")

    op_call = op_cls(
        (op_arg.basis, arg.basis),
        dtype,
        batch_dims,
        op_max_deg=np.int32(op_arg.basis.depth),
        arg_max_deg=np.int32(arg.basis.depth),
    )

    (result,) = op_call(op_arg.data, arg.data)

    return DenseFreeTensor(result, op_call.basis)


def st_adjoint_mul_derivative(
    op: ShuffleTensorT, arg: FreeTensorT, t_op: ShuffleTensorT, t_arg: FreeTensorT
) -> FreeTensorT:
    check_basis_compat(op.basis, arg.basis, t_op.basis, t_arg.basis)
    _get_and_check_batch_dims(op.data, arg.data, core_dims=1)

    t_result = st_adjoint_mul(op, t_arg) + st_adjoint_mul(t_op, arg)

    return t_result


def st_adjoint_mul_adjoint_derivative(
    op: ShuffleTensorT, arg: FreeTensorT, ct_result: ShuffleTensorT
) -> tuple[FreeTensorT, ShuffleTensorT]:
    check_basis_compat(op.basis, arg.basis, ct_result.basis)
    _get_and_check_batch_dims(op.data, arg.data, core_dims=1)

    ct_op = st_adjoint_mul(ct_result, arg)
    ct_arg = st_mul(op, ct_result)

    return ct_op, ct_arg


def _st_adjoint_mul_vjp_fwd(op, arg):
    result = st_adjoint_mul(op, arg)
    return result, (op, arg)


def _st_adjoint_mul_vjp_bwd(residuals, ct_result):
    op, arg = residuals
    ct_op, ct_arg = st_adjoint_mul_adjoint_derivative(op, arg, ct_result)
    return ct_op.data, ct_arg.data


st_adjoint_mul.defvjp(_st_adjoint_mul_vjp_fwd, _st_adjoint_mul_vjp_bwd)
