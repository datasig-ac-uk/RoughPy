from dataclasses import dataclass
from functools import partial
from typing import TypeVar

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from roughpy import compute as rpc
from .csc import csc_matvec


from roughpy_jax.ops import Operation

FreeTensorT = TypeVar('FreeTensorT')
ShuffleTensorT = TypeVar('ShuffleTensorT')
LieT = TypeVar('LieT')

# For exposition only
# class TensorBasis:
#     width: np.int32
#     depth: np.int32
#     degree_begin: np.ndarray[tuple[typing.Any], np.intp]
@partial(jax.tree_util.register_dataclass, data_fields=[], meta_fields=[])
class TensorBasis(rpc.TensorBasis):
    pass


# For exposition only
# class LieBasis:
#     width: np.int32
#     depth: np.int32
#     degree_begin: np.ndarray[tuple[typing.Any], np.intp]
#     data: np.ndarray[tuple[typing.Any, typing.Any], np.intp]
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


def _tensor_dataclass(cls):
    """
    Combined decorator for roughpy_jax tensor objects

    Registers dataclass and JAX data class with dynamic data and static basis
    """
    cls = dataclass(cls)
    return jax.tree_util.register_dataclass(
        cls,
        data_fields=["data"],
        meta_fields=["basis"]
    )


@_tensor_dataclass
class DenseFreeTensor:
    """
    Dense free tensor class built from basis and associated ndarray of data.
    """
    data: jnp.ndarray
    basis: TensorBasis


@_tensor_dataclass
class DenseShuffleTensor:
    """
    Dense shuffle tensor class built from basis and associated ndarray of data.
    """
    data: jnp.ndarray
    basis: TensorBasis


@_tensor_dataclass
class DenseLie:
    data: jnp.ndarray
    basis: LieBasis


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
                raise ValueError(f"Expecting jnp.float32 or jnp.float64 array for tensor {i}")

    for i, tensor in enumerate(other_tensors):
        if tensor.data.dtype != first_tensor.data.dtype:
            raise ValueError(f"Incompatible dtype between tensor 0 and tensor {i + 1}")


def _get_and_check_batch_dims(*arrays, core_dims=1):
    if not arrays:
        raise ValueError("expected at least one array")

    first, *rem = arrays

    n_dims = len(first.shape)
    if n_dims < core_dims:
        raise ValueError(f"array at index 0 has wrong number of dimensions, expected at least {core_dims}")

    batch_dims = first.shape[:-core_dims]
    n_batch_dims = len(batch_dims)

    for i, arr in enumerate(rem, start=1):
        if len(arr.shape) != n_dims:
            raise ValueError(f"mismatched number of dimensions at index {i}: "
                             f"expected {n_dims} but got {len(arr.shape)}")

        if arr.shape[:n_batch_dims] != batch_dims:
            raise ValueError(f"incompatible shape in argument at index {i}:"
                             f"expected batch shape {batch_dims} but got {arr.shape[:n_batch_dims]}")

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

    op = op_cls((a.basis, b.basis, c.basis), dtype, batch_dims,
                a_max_deg=a_max_deg,
                b_max_deg=min(a_max_deg, b.basis.depth),
                c_max_deg=min(a_max_deg, c.basis.depth),
                )

    out_data = op(a.data, b.data, c.data)

    return DenseFreeTensor(*out_data, op.basis)

    # FIXME commit conflict code incoming
    # # Use same basis convention as ft_fma in roughpy/compute
    # basis = a.basis
    # out_depth = a.basis.depth
    # lhs_depth = b.basis.depth
    # rhs_depth = c.basis.depth

    # call = jax.ffi.ffi_call(
    #     "cpu_dense_ft_fma",
    #     jax.ShapeDtypeStruct(a.data.shape, a.data.dtype)
    # )

    # out_data = call(
    #     a.data,
    #     b.data,
    #     c.data,
    #     width=np.int32(basis.width),
    #     depth=np.int32(basis.depth),
    #     degree_begin=basis.degree_begin,
    #     a_max_deg=np.int32(out_depth),
    #     b_max_deg=np.int32(lhs_depth),
    #     c_max_deg=np.int32(rhs_depth),
    #     b_min_deg=np.int32(0),
    #     c_min_deg=np.int32(0),
    # )

    # return DenseFreeTensor(out_data, basis)


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

    op = op_cls((a.basis, b.basis), dtype, batch_dims,
                out_max_deg=a.basis.depth,
                lhs_max_deg=min(a.basis.depth, a.basis.depth),
                rhs_max_deg=min(a.basis.depth, b.basis.depth))


    out_data = op(a.data, b.data)

    return DenseFreeTensor(*out_data, op.basis)

    # FIXME commit conflict code incoming
    # _check_tensor_dtype(a, b)
    # _check_basis_compat(a.basis, b.basis)

    # # Use same basis convention as ft_mul in roughpy/compute
    # basis = b.basis
    # out_depth = b.basis.depth
    # lhs_depth = a.basis.depth
    # rhs_depth = b.basis.depth

    # call = jax.ffi.ffi_call(
    #     "cpu_dense_ft_mul",
    #     jax.ShapeDtypeStruct(b.data.shape, b.data.dtype)
    # )

    # out_data = call(
    #     a.data,
    #     b.data,
    #     width=np.int32(basis.width),
    #     depth=np.int32(basis.depth),
    #     degree_begin=basis.degree_begin,
    #     out_max_deg=np.int32(out_depth),
    #     lhs_max_deg=np.int32(lhs_depth),
    #     rhs_max_deg=np.int32(rhs_depth),
    #     lhs_min_deg=np.int32(0),
    #     rhs_min_deg=np.int32(0),
    # )

    # return DenseFreeTensor(out_data, basis)


def antipode(a: FreeTensorT) -> FreeTensorT:
    """
    Antipode of a free tensor

    :param a: argument
    :return: new tensor with antipode of `a`
    """
    op_cls = Operation.get_operation("ft_antipode", "dense")
    batch_dims = _get_and_check_batch_dims(a.data, core_dims=1)

    basis = a.basis
    op = op_cls((basis,), a.data.dtype, batch_dims)

    out_data = op(a.data)

    return DenseFreeTensor(*out_data, basis)

    # FIXME commit conflict code incoming
    # out_data = call(
    #     a.data,
    #     width=np.int32(out_basis.width),
    #     depth=np.int32(out_basis.depth),
    #     degree_begin=out_basis.degree_begin,
    #     arg_max_deg=np.int32(a.basis.depth),
    #     no_sign=False
    # )

    # return DenseFreeTensor(out_data, out_basis)


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

    op = op_cls((a.basis, b.basis, c.basis), dtype, batch_dims,
                a_max_deg=a.basis.depth,
                b_max_deg=min(a.basis.depth, b.basis.depth),
                c_max_deg=min(b.basis.depth, c.basis.depth)
                )
    out_data = op(a.data, b.data, c.data)

    return DenseShuffleTensor(*out_data, op.basis)

    # FIXME commit conflict code incoming
    # out_data = call(
    #     a.data,
    #     b.data,
    #     c.data,
    #     width=np.int32(a.basis.width),
    #     depth=np.int32(a.basis.depth),
    #     degree_begin=a.basis.degree_begin,
    #     a_max_deg=np.int32(a.basis.depth),
    #     b_max_deg=np.int32(b.basis.depth),
    #     c_max_deg=np.int32(c.basis.depth),
    #     b_min_deg=np.int32(0),
    #     c_min_deg=np.int32(0),
    # )

    # return DenseShuffleTensor(out_data, a.basis)


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

    op = op_cls((lhs.basis, rhs.basis), dtype, batch_dims,
                out_max_deg=out_max_deg,
                lhs_max_deg=min(out_max_deg, lhs.basis.depth),
                rhs_max_deg=min(out_max_deg, rhs.basis.depth)
                )

    out_data = op(lhs.data, rhs.data)

    return DenseShuffleTensor(*out_data, op.basis)
    # call = jax.ffi.ffi_call(
    #     "cpu_dense_st_mul",
    #     jax.ShapeDtypeStruct(lhs.data.shape, lhs.data.dtype)
    # )

    # out_data = call(
    #     lhs.data,
    #     rhs.data,
    #     width=np.int32(lhs.basis.width),
    #     depth=np.int32(lhs.basis.depth),
    #     degree_begin=lhs.basis.degree_begin,
    #     out_max_deg=np.int32(lhs.basis.depth),
    #     lhs_max_deg=np.int32(lhs.basis.depth),
    #     rhs_max_deg=np.int32(rhs.basis.depth),
    #     lhs_min_deg=np.int32(0),
    #     rhs_min_deg=np.int32(0),
    # )

    # return DenseShuffleTensor(out_data, lhs.basis)


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
    batch_dims = x.data.shape[:-1]

    op_cls = Operation.get_operation("ft_exp", "dense")
    op = op_cls((out_basis, x.basis), dtype, batch_dims,
                out_max_deg=out_basis.depth)

    call = jax.ffi.ffi_call(
        "cpu_dense_ft_exp",
        jax.ShapeDtypeStruct(x.data.shape, x.data.dtype)
    )

    # FIXME convert to op

    out_data = call(
        x.data,
        width=np.int32(out_basis.width),
        depth=np.int32(out_basis.depth),
        degree_begin=out_basis.degree_begin,
        arg_max_deg=np.int32(x.basis.depth),
    )

    return DenseFreeTensor(out_data, out_basis)



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

    out_basis = out_basis or x.basis

    call = jax.ffi.ffi_call(
        "cpu_dense_ft_log",
        jax.ShapeDtypeStruct(x.data.shape, x.data.dtype)
    )

    # FIXME convert to op

    out_data = call(
        x.data,
        width=np.int32(out_basis.width),
        depth=np.int32(out_basis.depth),
        degree_begin=out_basis.degree_begin,
        arg_max_deg=np.int32(x.basis.depth)
    )

    return DenseFreeTensor(out_data, out_basis)


def ft_fmexp(multiplier: FreeTensorT, exponent: FreeTensorT, out_basis: TensorBasis | None = None) -> FreeTensorT:
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
    mul_depth = multiplier.basis.depth
    exp_depth = exponent.basis.depth

    call = jax.ffi.ffi_call(
        "cpu_dense_ft_fmexp",
        jax.ShapeDtypeStruct(multiplier.data.shape, multiplier.data.dtype)
    )

    # FIXME convert to op

    out_data = call(
        multiplier.data,
        exponent.data,
        width=np.int32(basis.width),
        depth=np.int32(basis.depth),
        degree_begin=out_basis.degree_begin,
        out_max_deg=np.int32(out_depth),
        mul_max_deg=np.int32(mul_depth),
        exp_max_deg=np.int32(exp_depth),
        mul_min_deg=np.int32(0),
        exp_min_deg=np.int32(0)
    )

    return DenseFreeTensor(out_data, basis)


def lie_to_tensor(arg: LieT, tensor_basis: TensorBasis | None = None, scale_factor=None) -> FreeTensorT:
    """
    Compute the embedding of a Lie algebra element as a free tensor.

    :param arg: Lie to embed into the tensor algebra
    :param tensor_basis: optional tensor basis to embed. Must have the same width as the Lie basis.
    :return: new FreeTensor containing the embedding of "arg"
    """

    l2t = arg.basis.get_l2t_matrix(arg.data.dtype)
    tensor_basis = tensor_basis or TensorBasis(arg.basis.width, arg.basis.depth)
    result = csc_matvec(l2t.data, l2t.indices, l2t.indptr, tensor_basis.size(), arg.data)
    if scale_factor:
        result = result * scale_factor

    return DenseFreeTensor(result, tensor_basis)


def tensor_to_lie(arg: FreeTensorT, lie_basis: LieBasis | None = None, scale_factor=None) -> LieT:
    """
    Project a free tensor onto the embedding of the Lie algebra in the tensor algebra.

    :param arg:
    :param lie_basis:
    :return:
    """
    lie_basis = lie_basis or LieBasis(arg.basis.width, arg.basis.depth)
    l2t = lie_basis.get_t2l_matrix(arg.data.dtype)

    result = csc_matvec(l2t.data, l2t.indices, l2t.indptr, lie_basis.size(), arg.data)
    if scale_factor:
        result = result * scale_factor

    return DenseLie(result, lie_basis)


def ft_adjoint_left_mul(op: FreeTensorT, arg: ShuffleTensorT) -> ShuffleTensorT:
    """
    Compute the adjoint action of left free-tensor multiplication on shuffles.

    Computes the adjoint action of the left multiplier operator L_a: b -> ab
    as an operator on shuffle tensors. That is, the shuffle tensor given
    by L_a^*(s) where * denotes adjoint.

    :param op: a in the notation above
    :param arg: The ShuffleTensor to be acted upon.
    :return: The result of the adjoint action as a ShuffleTensor.
    """

    out_basis = arg.basis
    op_max_deg = op.basis.depth
    arg_max_deg = arg.basis.depth

    call = jax.ffi.ffi_call(
        "cpu_dense_ft_adj_lmul",
        jax.ShapeDtypeStruct(arg.data.shape, arg.data.dtype)
    )

    # FIXME convert to op

    out_data = call(
        op.data,
        arg.data,
        width=np.int32(out_basis.width),
        depth=np.int32(out_basis.depth),
        degree_begin=out_basis.degree_begin,
        op_max_deg=np.int32(op_max_deg),
        arg_max_deg=np.int32(arg_max_deg)
    )

    return DenseShuffleTensor(out_data, out_basis)


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

    out_basis = arg.basis
    op_max_deg = op.basis.depth
    arg_max_deg = arg.basis.depth

    call = jax.ffi.ffi_call(
        "cpu_dense_ft_adj_rmul",
        jax.ShapeDtypeStruct(arg.data.shape, arg.data.dtype)
    )

    # FIXME convert to op

    out_data = call(
        op.data,
        arg.data,
        width=np.int32(out_basis.width),
        depth=np.int32(out_basis.depth),
        degree_begin=out_basis.degree_begin,
        op_max_deg=np.int32(op_max_deg),
        arg_max_deg=np.int32(arg_max_deg)
    )

    return DenseShuffleTensor(out_data, out_basis)
