"""
Free-standing compute functions that operate on plain NumPy arrays.

This provides the internal computational routines used by RoughPy using a NumPy based interface.
The functions and classes provided here can be used to implement custom algorithms that
require operations on free tensors, shuffle tensors, and Lie algebras.

This interface is designed to be stable and efficient, and interoperable with RoughPy main objects,
but is not directly tied the internals of RoughPy itself.

Functions that operate on free tensor data specifically are prefixed with "ft" (e.g. `ft_fma`).
Functions that operate on shuffle tensors are prefixed with "st" (e.g. `st_shuffle`).

"""
import typing
import numpy as np
import numpy.typing as npt

from dataclasses import dataclass

from . import _rpy_compute_internals as _internals

__all__ = []


def _api(version: str, *args, **kwargs):
    def deco(func_or_class):
        global __all__
        __all__.append(func_or_class.__name__)
        return func_or_class

    return deco


# Bases


# For exposition only
# class TensorBasis:
#     width: np.int32
#     depth: np.int32
#     degree_begin: np.ndarray[tuple[typing.Any], np.intp]


@_api("1.0.0")
class TensorBasis(_internals.TensorBasis):
    pass


# For exposition only
# class LieBasis:
#     width: np.int32
#     depth: np.int32
#     degree_begin: np.ndarray[tuple[typing.Any], np.intp]
#     data: np.ndarray[tuple[typing.Any, typing.Any], np.intp]


@_api("1.0.0")
class LieBasis(_internals.LieBasis):
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



SparseMatrix = _api("1.0.0")(_internals.SparseMatrix)

@_api("1.0.0")
@dataclass()
class DenseFreeTensor:
    data: np.ndarray[tuple[typing.Any], typing.Any]
    basis: TensorBasis


@_api("1.0.0")
@dataclass()
class DenseShuffleTensor:
    data: np.ndarray[tuple[typing.Any], typing.Any]
    basis: TensorBasis


@_api("1.0.0")
@dataclass()
class DenseLie:
    data: np.ndarray[tuple[typing.Any], typing.Any]
    basis: LieBasis


# Type aliases
FreeTensor = DenseFreeTensor
ShuffleTensor = DenseShuffleTensor
Lie = DenseLie


def _check_basis_compat(out_basis: TensorBasis, *other_bases: TensorBasis):
    out_width = out_basis.width

    for i, basis in enumerate(other_bases):
        if basis.width != out_width:
            raise ValueError(f"Incompatible width for basis {i}")


# Basic functions
@_api("1.0.0")
def ft_fma(a: FreeTensor, b: FreeTensor, c: FreeTensor, **kwargs):
    """
    Free tensor fused multiply-add.

    This function is equivalent to `a = b * c + a`.

    Currently only floating point scalars (np.float32, np.float64) are supported.

    :param a: output and first operand.
    :param b: left-hand operand
    :param c: right-hand operand
    """
    _check_basis_compat(a.basis, b.basis, c.basis)

    # In the future, we will need to handle alternative prototype tensors here.

    _internals.dense_ft_fma(a.data, b.data, c.data, b.basis, c.basis.depth)


@_api("1.0.0")
def ft_mul(a: FreeTensor, b: FreeTensor, **kwargs) -> FreeTensor:
    """
    Free tensor multiply.

    Currently only floating point scalars (np.float32, np.float64) are supported.

    :param a: left-hand operand
    :param b: right-hand operand
    :return: product of `a` and `b`
    """
    _check_basis_compat(a.basis, b.basis)

    result = np.zeros_like(a.data)

    _internals.dense_ft_fma(result, a.data, b.data, a.basis, rhs_depth=b.basis.depth)

    return DenseFreeTensor(result, a.basis)


@_api("1.0.0")
def ft_inplace_mul(a: FreeTensor, b: FreeTensor, **kwargs):
    """
    Free tensor in-place multiplication.

    This function is equivalent to `a = a * b`.

    Currently only floating point scalars (np.float32, np.float64) are supported.

    :param a: output and first operand.
    :param b: right-hand operand
    """
    _check_basis_compat(a.basis, b.basis)

    # In the future, we need to handle alternative prototype tensors here.

    _internals.dense_ft_inplace_mul(a.data, b.data, a.basis, b.basis.depth)


@_api("1.0.0")
def antipode(a: FreeTensor, **kwargs) -> FreeTensor:
    """
    Antipode of a free tensor.


    :param a: argument
    :return: new tensor with antipode of `a`
    """

    result = np.zeros_like(a.data)
    _internals.dense_ft_antipode(result, a.data, a.basis)

    return FreeTensor(result, a.basis)


def st_fma(*args, **kwargs):
    ...


def st_inplace_mul(*args, **kwargs):
    ...


def lie_to_tensor(*args, **kwargs):
    ...


def tensor_to_lie(*args, **kwargs):
    ...


def ft_exp(x: FreeTensor, out_basis: TensorBasis | None = None) -> FreeTensor:

    """
    Exponential of a free tensor.

    :param x: argument
    :param out_basis: optional output basis. If not specified, the same basis as `x` is used.
    :return: tensor exponential of `x`
    """

    out_basis = out_basis or x.basis

    _check_basis_compat(out_basis, x.basis)

    dtype = x.data.dtype
    if dtype not in (np.float32, np.float64):
        raise ValueError(f"Unsupported dtype {dtype}")

    shape = (*x.data.shape[:-1], out_basis.size())

    result = np.zeros(shape, dtype=dtype)

    _internals.dense_ft_exp(result, x.data, out_basis)

    return FreeTensor(result, out_basis)


def ft_log(x: FreeTensor, out_basis: TensorBasis | None = None) -> FreeTensor:
    """
    Logarithm of a free tensor.


    :param x: tensor to take logarithm of
    :param out_basis: optional output basis. If not specified, the same basis as `x` is used.
    :return: tensor logarithm of `x`
    """

    out_basis = out_basis or x.basis
    _check_basis_compat(out_basis, x.basis)

    dtype = x.data.dtype
    if dtype not in (np.float32, np.float64):
        raise ValueError(f"Unsupported dtype {dtype}")

    shape = (*x.data.shape[:-1], out_basis.size())

    result = np.zeros(shape, dtype=dtype)

    _internals.dense_ft_log(result, x.data, out_basis)

    return FreeTensor(result, out_basis)


def ft_fmexp(multiplier: FreeTensor, exponent: FreeTensor, out_basis: TensorBasis | None = None) -> FreeTensor:
    """
    Fused multiply-exponential of two free tensors.

    Computes the fused product A*exp(X) where A is `multiplier` and X is `exponent`.

    :param multiplier: Multiplier free tensor
    :param exponent: Free tensor to exponential
    :param out_basis: Optional output basis. If not specified, the same basis as `multiplier` is used.
    :return: The result of fused multiply-exponential of `multiplier` and `exponent`
    """

    out_basis = out_basis or multiplier.basis
    _check_basis_compat(out_basis, multiplier.basis, exponent.basis)

    dtype = multiplier.data.dtype
    if dtype not in (np.float32, np.float64):
        raise ValueError(f"Unsupported dtype {dtype}")

    shape = (*multiplier.data.shape[:-1], out_basis.size())

    result = np.zeros(shape, dtype=dtype)

    _internals.dense_ft_fmexp(result, multiplier.data, exponent.data, out_basis)

    return FreeTensor(result, out_basis)



@_api("1.0.0")
def ft_adjoint_left_mul(op: FreeTensor, arg: ShuffleTensor) -> ShuffleTensor:
    """
    Compute the adjoint of a free tensor left-multiplication.

    The operator L_A: T -> T defined by L_A(X) = A * X (where * denotes
    free tensor multiplication) is a well-defined linear operator on the
    free tensor algebra. The adjoint of this function L_A* is a linear
    operator on the shuffle algebra. This operator aggregates the
    coefficients of S according to their prefix in A.

    :param op: The operand of the left multiplication L_A
    :param arg: The shuffle tensor argument on which to apply the adjoint
    :return: The result of L_A*(S)
    """

    _check_basis_compat(op.basis, arg.basis)

    result = np.zeros_like(arg.data)

    _internals.dense_ft_adjoint_left_mul(
        result, op.data, arg.data, op.basis, arg.basis.depth, op.basis.depth, arg.basis.depth)

    return ShuffleTensor(result, arg.basis)

