from dataclasses import dataclass
from functools import partial, partialmethod
from typing import TypeVar, Callable

import jax
import jax.numpy as jnp
import numpy as np

from roughpy import compute as rpc

from roughpy_jax.ops import Operation

AlgebraT = TypeVar("AlgebraT")
FreeTensorT = TypeVar("FreeTensorT")
ShuffleTensorT = TypeVar("ShuffleTensorT")
LieT = TypeVar("LieT")


# For exposition only
# class TensorBasis:
#     width: np.int32
#     depth: np.int32
#     degree_begin: np.ndarray[np.int64.dtype]
@partial(jax.tree_util.register_dataclass, data_fields=[], meta_fields=[])
class TensorBasis(rpc.TensorBasis):
    pass


# For exposition only
# class LieBasis:
#     width: np.int32
#     depth: np.int32
#     degree_begin: np.ndarray[np.int64.dtype]
@partial(jax.tree_util.register_dataclass, data_fields=[], meta_fields=[])
class LieBasis(rpc.LieBasis):
    """
    An instance of a Hall basis for the Lie algebra.

    A Hall basis is indexed by integer keys k > 0. To each key there is an
    associated pair of parents (a, b) where a and b are both keys belonging
    to the Hall basis. The exceptions are the "letters", which are those keys
    k for which the parents are (0, k). For convenience, we usually add a null
    element to the basis at key 0 and with parents (0, 0), which serves to
    offset elements correctly. However, this is not a valid key for the vectors
    and thus the key to index map subtracts 1 from the key to obtain the
    position in the vector.

    The default constructor requires only width and depth and constructs a
    Hall set greedily, minimizing the degree of the left parent. For instance,
    for width 2 and depth 4, the basis contains 5 keys 1 -> (0, 1), 2 -> (0, 2),
    3 -> (1, 2) (which represents the bracket [1,2]), 4 -> (1, 3) ([1,[1,2]]),
    and 5 -> (2, 3) ([2,[1,2]]).

    This implementation is designed to be flexible as to the exact contents of
    the Hall set, provided it is given in the format described above. The basis
    must also be ordered by degree, so elements of degree k must appear
    sequentially and between elements of degree k - 1 and degree k + 1 (if such
    elements exist).
    """

    pass


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


def _algebra_scalar_multiply(a: AlgebraT, s: jax.typing.ArrayLike) -> AlgebraT:
    cls = type(a)
    basis = a.basis

    result_data = jnp.dot(a.data, s)

    return cls(result_data, basis)


def _algebra___array__(self, dtype=None, copy=None) -> np.ndarray:
    return self.data.__array__(dtype=dtype, copy=copy)


def _tensor_dataclass(cls):
    """
    Combined decorator for roughpy_jax tensor objects

    Registers dataclass and JAX data class with dynamic data and static basis
    """
    cls = dataclass(cls)

    cls.__array__ = _algebra___array__

    cls.__add__ = partialmethod(_algebra_add, impl=jnp.add)
    cls.__sub__ = partialmethod(_algebra_add, impl=jnp.subtract)

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


def _check_basis_compat(first_basis: TensorBasis, *other_bases: TensorBasis):
    for i, basis in enumerate(other_bases):
        if basis.width != first_basis.width:
            raise ValueError(f"Incompatible width between basis 0 and basis {i + 1}")


def _check_tensor_dtype(first_tensor: FreeTensor, *other_tensors: FreeTensor):
    for i, ft in enumerate([first_tensor] + list(other_tensors)):
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
) -> FreeTensorT: ...


def ft_fma_adjoint_derivative(
    a: FreeTensorT,
    b: FreeTensorT,
    c: FreeTensorT,
    ct_result: ShuffleTensorT,
) -> tuple[ShuffleTensorT, ShuffleTensorT, ShuffleTensorT]: ...


def ft_mul(a: FreeTensorT, b: FreeTensorT) -> FreeTensorT:
    """
    Free tensor multiply

    This function is equivalent to `a * b`.
    Supports float 32 or 64 but all data buffers must have matching type.
    The basis is taken from `b`.

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
        out_max_deg=np.int32(a.basis.depth),
        lhs_max_deg=np.int32(min(a.basis.depth, a.basis.depth)),
        rhs_max_deg=np.int32(min(a.basis.depth, b.basis.depth)),
        lhs_min_deg=np.int32(0),
        rhs_min_deg=np.int32(0),
    )

    out_data = op(a.data, b.data)

    return DenseFreeTensor(*out_data, op.basis)


def ft_mul_derivative(
    a: FreeTensorT, b: FreeTensorT, t_a: FreeTensorT, t_b: FreeTensorT
) -> FreeTensorT: ...


def ft_mul_adjoint_derivative(
    a: FreeTensorT, b: FreeTensorT, ct_result: ShuffleTensorT
) -> tuple[ShuffleTensorT, ShuffleTensorT]: ...


@jax.custom_vjp
def antipode(a: AlgebraT) -> AlgebraT:
    """
    Antipode of a free tensor

    :param a: argument
    :return: new tensor with antipode of `a`
    """
    op_cls = Operation.get_operation("ft_antipode", "dense")
    batch_dims = _get_and_check_batch_dims(a.data, core_dims=1)

    out_class = a.__class__
    out_basis = a.basis

    op = op_cls(
        (out_basis,),
        a.data.dtype,
        batch_dims,
        arg_max_deg=np.int32(out_basis.depth),
        no_sign=False,
    )

    out_data = op(a.data)

    return out_class(*out_data, out_basis)


def antipode_derivative(a: FreeTensorT, t_a: FreeTensorT) -> FreeTensorT:
    """
    Antipode derivative of free tensor peterbation `t_a` at `a`

    This operation is linear, with the derivative being independent of
    the argument, computated as the antipode of the tangent. This is
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
    a, = residuals

    ct_result = DenseShuffleTensor(ct_result_data.data, ct_result_data.basis)
    ct_antipode = antipode_adjoint_derivative(
        a, ct_result
    )

    return (ct_antipode.data,)


antipode.defvjp(_antipode_vjp_fwd, _antipode_vjp_bwd)


def st_fma(a: ShuffleTensorT, b: ShuffleTensorT, c: ShuffleTensorT) -> ShuffleTensorT:
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
) -> ShuffleTensorT: ...


def st_fma_adjoint_derivative(
    a: ShuffleTensorT,
    b: ShuffleTensorT,
    c: ShuffleTensorT,
    ct_result: FreeTensorT,
) -> tuple[FreeTensorT, FreeTensorT, FreeTensorT]: ...


def st_mul(lhs: ShuffleTensorT, rhs: ShuffleTensorT) -> ShuffleTensorT:
    """
    Shuffle tensor product

    This function is equivalent to `lhs & rhs`.
    Supports float 32 or 64 but all data buffers must have matching type.
    The result basis is taken from `lhs`.

    :param lhs: left-hand operand
    :param rhs: right-hand operand
    :return: the shuffle product of lhs and rhs
    """
    dtype = jnp.result_type(lhs.data.dtype, rhs.data.dtype)
    batch_dims = _get_and_check_batch_dims(lhs.data, rhs.data, core_dims=1)

    op_cls = Operation.get_operation("st_mul", "dense")
    out_max_deg = lhs.basis.depth

    op = op_cls(
        (lhs.basis, rhs.basis),
        dtype,
        batch_dims,
        out_max_deg=np.int32(out_max_deg),
        lhs_max_deg=np.int32(min(out_max_deg, lhs.basis.depth)),
        rhs_max_deg=np.int32(min(out_max_deg, rhs.basis.depth)),
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
) -> ShuffleTensorT: ...


def st_mul_adjoint_derivative(
    lhs: ShuffleTensorT, rhs: ShuffleTensorT, ct_result: FreeTensorT
) -> tuple[FreeTensorT, FreeTensorT]: ...


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

    out_basis = out_basis or x.basis

    op_cls = Operation.get_operation("ft_exp", "dense")
    op = op_cls(
        (out_basis, x.basis),
        dtype,
        x.batch_shape,
        arg_max_deg=np.int32(out_basis.depth),
    )

    out_data = op(x.data)

    return DenseFreeTensor(*out_data, out_basis)


def ft_exp_derivative(
    x: FreeTensorT,
    t_x: FreeTensorT,
) -> FreeTensorT: ...


def ft_exp_adjoint_derivative(
    x: FreeTensorT,
    ct_result: ShuffleTensorT,
) -> tuple[ShuffleTensorT]: ...


def ft_log(x: FreeTensorT, out_basis: TensorBasis | None = None) -> FreeTensorT:
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
    dtype = x.data.dtype

    out_basis = out_basis or x.basis

    op_cls = Operation.get_operation("ft_log", "dense")
    op = op_cls(
        (out_basis, x.basis),
        dtype,
        x.batch_shape,
        arg_max_deg=np.int32(out_basis.depth),
    )

    out_data = op(x.data)

    return DenseFreeTensor(*out_data, out_basis)


def ft_log_derivative(
    x: FreeTensorT,
    t_x: FreeTensorT,
) -> FreeTensorT: ...


def ft_log_adjoint_derivative(
    x: FreeTensorT,
    ct_result: ShuffleTensorT,
) -> tuple[ShuffleTensorT]: ...


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

    out_basis = out_basis or multiplier.basis
    _check_basis_compat(out_basis, multiplier.basis, exponent.basis)

    batch_dims = _get_and_check_batch_dims(multiplier.data, exponent.data, core_dims=1)

    out_depth = multiplier.basis.depth
    mul_depth = multiplier.basis.depth
    exp_depth = exponent.basis.depth

    op_cls = Operation.get_operation("ft_fmexp", "dense")
    op = op_cls(
        (out_basis, multiplier.basis, exponent.basis),
        dtype,
        batch_dims,
        out_max_deg=np.int32(out_depth),
        mul_max_deg=np.int32(mul_depth),
        exp_max_deg=np.int32(exp_depth),
        mul_min_deg=np.int32(0),
        exp_min_deg=np.int32(0),
    )

    out_data = op(multiplier.data, exponent.data)

    return DenseFreeTensor(*out_data, out_basis)


def ft_fmexp_derivative(
    multiplier: FreeTensorT,
    exponent: FreeTensorT,
    t_multiplier: FreeTensorT,
    t_exponent: FreeTensorT,
) -> FreeTensorT:
    _check_basis_compat(
        multiplier.basis, exponent.basis, t_multiplier.basis, t_exponent.basis
    )
    _get_and_check_batch_dims(
        multiplier.data, exponent.data, t_multiplier.data, t_exponent.data, core_dims=1
    )

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
    _check_basis_compat(multiplier.basis, exponent.basis, ct_result.basis)
    _get_and_check_batch_dims(
        multiplier.data, exponent.data, ct_result.data, core_dims=1
    )

    tensor_type = type(multiplier)
    ct_type = type(ct_result)

    basis = multiplier.basis
    depth = basis.depth

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


def lie_to_tensor(
    arg: LieT, tensor_basis: TensorBasis | None = None, scale_factor=None
) -> FreeTensorT:
    """
    Compute the embedding of a Lie algebra element as a free tensor.

    :param arg: Lie to embed into the tensor algebra
    :param tensor_basis: optional tensor basis to embed. Must have the same width as the Lie basis.
    :return: new FreeTensor containing the embedding of "arg"
    """
    _check_tensor_dtype(arg)
    dtype = arg.data.dtype
    out_basis = arg.basis

    tensor_basis = tensor_basis or out_basis

    op_cls = Operation.get_operation("lie_to_tensor", "dense")
    op = op_cls(
        (out_basis, tensor_basis),
        dtype,
        arg.batch_shape,
        scale_factor=scale_factor,
    )

    out_data = op(arg.data)
    return DenseFreeTensor(*out_data, out_basis)


def lie_to_tensor_derivative(
    arg: LieT,
    t_arg: LieT,
    scale_factor=None,
) -> FreeTensorT: ...


def lie_to_tensor_adjoint_derivative(
    arg: LieT,
    ct_result: ShuffleTensorT,
    scale_factor=None,
) -> tuple[LieT]: ...


def tensor_to_lie(
    arg: FreeTensorT, lie_basis: LieBasis | None = None, scale_factor=None
) -> LieT:
    """
    Project a free tensor onto the embedding of the Lie algebra in the tensor algebra.

    :param arg:
    :param lie_basis:
    :return:
    """
    _check_tensor_dtype(arg)
    dtype = arg.data.dtype
    out_basis = arg.basis

    lie_basis = lie_basis or out_basis

    op_cls = Operation.get_operation("tensor_to_lie", "dense")
    op = op_cls(
        (out_basis, lie_basis),
        dtype,
        arg.batch_shape,
        scale_factor=scale_factor,
    )

    out_data = op(arg.data)
    return DenseLie(*out_data, out_basis)


def tensor_to_lie_derivative(
    arg: FreeTensorT,
    t_arg: FreeTensorT | None = None,
    scale_factor=None,
) -> LieT: ...


def tensor_to_lie_adjoint_derivative(
    arg: FreeTensorT,
    ct_result: LieT,
    scale_factor=None,
) -> tuple[ShuffleTensorT]: ...


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

    out_basis = arg.basis
    _check_basis_compat(out_basis, op.basis)

    batch_dims = _get_and_check_batch_dims(op.data, arg.data, core_dims=1)

    op_max_deg = op.basis.depth
    arg_max_deg = arg.basis.depth

    op_cls = Operation.get_operation("ft_adj_lmul", "dense")
    op_call = op_cls(
        (out_basis, op.basis),
        dtype,
        batch_dims,
        degree_begin=out_basis.degree_begin,
        op_max_deg=np.int32(op_max_deg),
        arg_max_deg=np.int32(arg_max_deg),
    )

    out_data = op_call(op.data, arg.data)

    return DenseShuffleTensor(*out_data, out_basis)


def ft_adjoint_left_mul_derivative(
    op: FreeTensorT,
    arg: ShuffleTensorT,
    t_op: FreeTensorT,
    t_arg: ShuffleTensorT,
) -> ShuffleTensorT: ...


def ft_adjoint_left_mul_adjoint_derivative(
    op: FreeTensorT, arg: ShuffleTensorT, ct_result: FreeTensorT
) -> tuple[ShuffleTensorT, FreeTensorT]: ...


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

    out_basis = arg.basis
    _check_basis_compat(out_basis, op.basis)

    batch_dims = _get_and_check_batch_dims(op.data, arg.data, core_dims=1)

    op_max_deg = op.basis.depth
    arg_max_deg = arg.basis.depth

    op_cls = Operation.get_operation("ft_adj_rmul", "dense")
    op_call = op_cls(
        (out_basis, op.basis),
        dtype,
        batch_dims,
        degree_begin=out_basis.degree_begin,
        op_max_deg=np.int32(op_max_deg),
        arg_max_deg=np.int32(arg_max_deg),
    )

    out_data = op_call(op.data, arg.data)

    return DenseShuffleTensor(*out_data, out_basis)


def ft_adjoint_right_mul_derivative(
    op: FreeTensorT,
    arg: ShuffleTensorT,
    t_op: FreeTensorT,
    t_arg: ShuffleTensorT,
) -> ShuffleTensorT: ...


def ft_adjoint_right_mul_adjoint_derivative(
    op: FreeTensorT, arg: ShuffleTensorT, ct_result: FreeTensorT
) -> tuple[ShuffleTensorT, FreeTensorT]: ...


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

    _check_basis_compat(functional.basis, functional.basis)
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
) -> jax.Array: ...


def tensor_pairing_adjoint_derivative(
    functional: ShuffleTensorT,
    argument: FreeTensorT,
    ct_result: jax.Array,
) -> tuple[FreeTensorT, ShuffleTensorT]: ...
