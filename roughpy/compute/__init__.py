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
    pass



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
    _check_basis_compat(a.basis, b.basis, b.basis)

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




def ft_exp(x: FreeTensor) -> FreeTensor:
    """
    Exponential of a free tensor.

    :param x: argument
    :return: tensor exponential of `x`
    """

    result = np.zeros_like(x.data)

    # This is a pure python implementation that will be
    # replaced by a C++ implementation once available.
    depth = x.basis.depth
    result[0] = 1

    for deg in range(0, depth):
        z = x.data / (depth - deg)
        _internals.dense_ft_inplace_mul(result, z, x.basis)

        result[0] += 1

    return FreeTensor(result, x.basis)
