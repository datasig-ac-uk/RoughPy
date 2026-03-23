from dataclasses import dataclass
from functools import partial, partialmethod
from typing import TypeVar, Callable, Type, Any, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

from roughpy_jax.ops import Operation
from roughpy_jax.dense_algebra import (
    DenseAlgebra,
    DenseTensor,
    get_common_batch_shape,
    broadcast_to_batch_shape,
    zero_like,
    identity_like,
)

from .bases import (
    TensorBasis,
    LieBasis,
    check_basis_compat,
    to_lie_basis,
    to_tensor_basis,
)
from .compressed import csr_matvec

T = TypeVar("T")
AlgebraT = TypeVar("AlgebraT")
FreeTensorT = TypeVar("FreeTensorT")
ShuffleTensorT = TypeVar("ShuffleTensorT")
LieT = TypeVar("LieT")


@jax.tree_util.register_pytree_node_class
class DenseFreeTensor(DenseTensor):

    def __matmul__(self, other):
        if isinstance(other, FreeTensor):
            return ft_mul(self, other)
        return NotImplemented


@jax.tree_util.register_pytree_node_class
class DenseShuffleTensor(DenseTensor):

    def __matmul__(self, other):
        if isinstance(other, ShuffleTensor):
            return st_mul(self, other)
        return NotImplemented


@jax.tree_util.register_pytree_node_class
class DenseLie(DenseAlgebra[LieBasis]):
    pass


"""
Tensor aliases. Tensors are assumed to be dense without prefix

TODO: These should be replaced by type aliases later, and we should
      look at the types of the inputs to determine the correct type
      for the output in each case. For now, we assume everything is
      dense, but we should nonetheless do this replacement soon.
"""
FreeTensor: TypeAlias = DenseFreeTensor
ShuffleTensor: TypeAlias = DenseShuffleTensor
Lie: TypeAlias = DenseLie


def _remove_unit_term(tensor: AlgebraT) -> AlgebraT:
    if not isinstance(tensor.basis, TensorBasis):
        raise TypeError(f"object must be a tensor")

    new_data = tensor.data.at[..., 0].set(0)
    return type(tensor)(new_data, tensor.basis)


def as_free_tensor(tensor: FreeTensor | ShuffleTensor) -> FreeTensor:
    """
    Converts a given tensor to a FreeTensor instance if it is not already of that type.

    :param tensor: The tensor to be converted. It can either be a FreeTensor or
        a ShuffleTensor.
    :return: A FreeTensor object derived from the input tensor.
    """
    if isinstance(tensor, FreeTensor):
        return tensor
    return FreeTensor(tensor.data, tensor.basis)


def as_shuffle_tensor(tensor: FreeTensor | ShuffleTensor) -> ShuffleTensor:
    """
    Convert a tensor into a ShuffleTensor instance if it is not already of that type.

    This function ensures that a given tensor is returned as a `ShuffleTensor`
    object. If the input tensor is already a `ShuffleTensor`, it is returned
    directly. Otherwise, a new `ShuffleTensor` is created using the data and basis
    of the input tensor.

    :param tensor: The input tensor to be converted. Can be an instance of either
        `FreeTensor` or `ShuffleTensor`.
    :return: An instance of `ShuffleTensor`. If the input was already a
        `ShuffleTensor`, it returns the same object. Otherwise, it constructs and
        returns a new `ShuffleTensor`.
    """
    if isinstance(tensor, ShuffleTensor):
        return tensor
    return ShuffleTensor(tensor.data, tensor.basis)


def to_jax_cotangent(
    primal_cls: type[DenseAlgebra], cotangent: DenseAlgebra
) -> DenseAlgebra:
    """
    Convert a mathematical cotangent into the pytree representation expected by JAX.

    JAX's ``custom_vjp`` interface does not model cotangents as arbitrary dual-space
    objects. Instead, a backward rule must return cotangents whose pytree structure
    matches the corresponding primal argument. For these algebra types that is not,
    in general, the same thing as the mathematically correct cotangent object:
    the cotangent of a free tensor is naturally a shuffle tensor, and vice versa.

    This helper makes that mismatch explicit. It takes a cotangent expressed in the
    mathematically correct dual algebra and re-encodes it using the concrete pytree
    type JAX expects for ``primal``. The returned object is therefore a JAX-facing
    cotangent representation, not a statement that the cotangent space is actually
    the same algebra as the primal.

    :param primal_cls: The concrete primal algebra type whose pytree structure
        JAX expects in the backward return value.
    :param cotangent: The cotangent expressed in the mathematically appropriate
        dual algebra.
    :return: A cotangent encoded using ``primal_cls`` and the basis already
        attached to ``cotangent`` so that it matches the pytree structure
        required by JAX.
    """
    return primal_cls(cotangent.data, cotangent.basis)


def from_jax_cotangent(
    primal_cls: type[DenseAlgebra], cotangent: DenseAlgebra | jax.Array, basis
) -> DenseAlgebra:
    """
    Convert a JAX cotangent representation back into the mathematically correct type.

    When JAX invokes a custom VJP backward rule, the cotangent argument may be
    reconstructed using the primal pytree structure rather than the true dual
    algebra type. This helper reverses the representation coercion performed by
    :func:`to_jax_cotangent`, recovering the cotangent algebra associated with
    ``primal`` before the backward rule performs any algebraic operations.

    For roughpy-jax's truncated algebras this means:
    ``FreeTensor -> ShuffleTensor``, ``ShuffleTensor -> FreeTensor``, and
    ``Lie -> Lie``.

    :param primal_cls: The concrete primal algebra type whose dual-space
        cotangent type is required.
    :param cotangent: The cotangent value as reconstructed by JAX, either as a raw
        array leaf or as an algebra object with primal-shaped pytree structure.
    :param basis: Basis metadata to attach to the recovered cotangent.
    :return: The cotangent reinterpreted in the mathematically correct dual algebra.
    """
    if isinstance(cotangent, DenseAlgebra):
        data = cotangent.data
    else:
        data = cotangent

    return primal_cls.DualVector(data, basis)


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
    batch_dims = get_common_batch_shape(a, b, c)

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
    batch_dims = get_common_batch_shape(a, b)

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
    batch_dims = get_common_batch_shape(a)

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
    batch_dims = get_common_batch_shape(a, b, c)
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
    get_common_batch_shape(a, b, c, t_a, t_b, t_c)

    return t_a + st_mul_derivative(b, c, t_b, t_c)


def st_fma_adjoint_derivative(
    a: ShuffleTensorT,
    b: ShuffleTensorT,
    c: ShuffleTensorT,
    ct_result: FreeTensorT,
) -> tuple[FreeTensorT, FreeTensorT, FreeTensorT]:
    check_basis_compat(a.basis, b.basis, c.basis, ct_result.basis)
    get_common_batch_shape(a, b, c, ct_result)

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
    batch_dims = get_common_batch_shape(lhs, rhs)

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
    get_common_batch_shape(lhs, rhs, t_lhs, t_rhs)

    t_result = st_mul(lhs, t_rhs) + st_mul(t_lhs, rhs)
    return t_result


def st_mul_adjoint_derivative(
    lhs: ShuffleTensorT, rhs: ShuffleTensorT, ct_result: FreeTensorT
) -> tuple[FreeTensorT, FreeTensorT]:
    check_basis_compat(lhs.basis, rhs.basis, ct_result.basis)
    get_common_batch_shape(lhs, rhs, ct_result)

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
    If `out_basis` is not specified, the same basis as `x` is used.

    :param x: argument
    :param out_basis: optional output basis.
    :return: tensor exponential of `x`
    """
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
    batch_dims = get_common_batch_shape(x, t_x)
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
    batch_dims = get_common_batch_shape(x, ct_result)
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

    If `out_basis` is not specified, the same basis as `x` is used.

    :param x: argument
    :param out_basis: optional output basis.
    :return: tensor logarithm of `x`
    """
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
    batch_dims = get_common_batch_shape(x, t_x)
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
    batch_dims = get_common_batch_shape(x, ct_result)
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
    If `out_basis` is not specified, the same basis as `multiplier` is used.

    :param multiplier: Multiplier free tensor
    :param exponent: Free tensor to exponential
    :param out_basis: Optional output basis. If not specified, the same basis as `multiplier` is used.
    :return: Resulting fused multiply-exponential of `multiplier` and `exponent`
    """
    dtype = jnp.result_type(multiplier.dtype, exponent.dtype)
    batch_dims = get_common_batch_shape(multiplier, exponent)

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
    get_common_batch_shape(multiplier, exponent, t_multiplier, t_exponent)

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
    get_common_batch_shape(multiplier, exponent, ct_result)

    tensor_type = type(multiplier)
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

    batch_dims = get_common_batch_shape(op, arg)

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
    get_common_batch_shape(op, arg, t_op, t_arg)

    t_result = ft_adjoint_left_mul(t_op, arg) + ft_adjoint_left_mul(op, t_arg)

    return t_result


def ft_adjoint_left_mul_adjoint_derivative(
    op: FreeTensorT, arg: ShuffleTensorT, ct_result: FreeTensorT
) -> tuple[ShuffleTensorT, FreeTensorT]:
    check_basis_compat(op.basis, arg.basis, ct_result.basis)
    get_common_batch_shape(op, arg, ct_result)

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

    batch_dims = get_common_batch_shape(op, arg)

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
    get_common_batch_shape(op, arg, t_op, t_arg)

    t_result = ft_adjoint_right_mul(t_op, arg) + ft_adjoint_right_mul(op, t_arg)
    return t_result


def ft_adjoint_right_mul_adjoint_derivative(
    op: FreeTensorT, arg: ShuffleTensorT, ct_result: FreeTensorT
) -> tuple[ShuffleTensorT, FreeTensorT]:
    check_basis_compat(op.basis, arg.basis, ct_result.basis)
    get_common_batch_shape(op, arg, ct_result)

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
    batch_dims = get_common_batch_shape(functional, argument)

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
    get_common_batch_shape(functional, t_functional, t_argument)

    x = tensor_pairing(functional, t_argument)
    y = tensor_pairing(t_functional, argument)
    return x + y


def _reshape_pairing_cotangent(
    ct_result: jax.Array, batch_dims: tuple[int, ...]
) -> jax.Array:
    ct_shape = ct_result.shape
    if len(ct_shape) > len(batch_dims) or ct_shape != batch_dims[: len(ct_shape)]:
        raise ValueError(
            f"incompatible shapes: {ct_shape} and {batch_dims[:len(ct_shape)]}"
        )

    new_shape = ct_shape + (1,) * (len(batch_dims) - len(ct_shape)) + (1,)
    return jnp.reshape(ct_result, new_shape)


def tensor_pairing_adjoint_derivative(
    functional: ShuffleTensorT,
    argument: FreeTensorT,
    ct_result: jax.Array,
) -> tuple[FreeTensorT, ShuffleTensorT]:
    check_basis_compat(functional.basis, argument.basis)
    batch_dims = get_common_batch_shape(functional, argument)

    ext_ct = broadcast_to_batch_shape(ct_result, batch_dims)

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
    batch_dims = get_common_batch_shape(functional, argument)

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
    get_common_batch_shape(functional, argument, t_functional, t_argument)

    x = lie_pairing(functional, t_argument)
    y = lie_pairing(t_functional, argument)
    return x + y


def lie_pairing_adjoint_derivative(
    functional: LieT, argument: LieT, ct_result: jax.Array
) -> tuple[LieT, LieT]:
    check_basis_compat(functional.basis, argument.basis)
    batch_dims = get_common_batch_shape(functional, argument)

    ext_ct = broadcast_to_batch_shape(ct_result, batch_dims)

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
    batch_dims = get_common_batch_shape(op_arg, arg)

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
    get_common_batch_shape(op, arg, t_op, t_arg)

    t_result = st_adjoint_mul(op, t_arg) + st_adjoint_mul(t_op, arg)

    return t_result


def st_adjoint_mul_adjoint_derivative(
    op: ShuffleTensorT, arg: FreeTensorT, ct_result: ShuffleTensorT
) -> tuple[FreeTensorT, ShuffleTensorT]:
    check_basis_compat(op.basis, arg.basis, ct_result.basis)
    get_common_batch_shape(op, arg, ct_result)

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
