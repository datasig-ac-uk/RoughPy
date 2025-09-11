import jax
import jax.numpy as jnp
import jax.lax as lax

from dataclasses import dataclass
from typing import NamedTuple


try:
    from . import _rpy_jax_internals
except ImportError as e:
    _rpy_jax_internals = None
    raise ImportError("RoughPy JAX CPU backend is not installed correctly") from e
else:
    # FIXME create using register
    jax.ffi.register_ffi_target(
        "cpu_dense_ft_fma",
        _rpy_jax_internals.cpu_dense_ft_fma(),
        "cpu"
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
    degree_begin: jnp.ndarray

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

        self.degree_begin = degree_begin

    def size(self):
        return self.degree_begin[self.depth + 1]


class DenseFreeTensor(NamedTuple):
    """
    Dense free tensor class built from basis and associated ndarray of data.
    """
    data: jnp.ndarray
    basis: TensorBasis


def _check_basis_compat(out_basis: TensorBasis, *other_bases: TensorBasis):
    out_width = out_basis.width

    for i, basis in enumerate(other_bases):
        if basis.width != out_width:
            raise ValueError(f"Incompatible width for basis {i}")


def dense_ft_fma(
    a: DenseFreeTensor,
    b: DenseFreeTensor,
    c: DenseFreeTensor
) -> DenseFreeTensor:
    """
    Free tensor fused multiply-add.

    This function is equivalent to `d = b * c + a`.

    Currently only floating point scalars (np.float32) are supported.

    The roughpy compute equivalent is ternary and mutates the first operand a
    to be the output. In JAX arrays are immutable so this version differs by
    copying `a` and returing a new result with its size and shape.

    The basis is taken from `b`.

    :param a: addition operand
    :param b: left-hand multiply operand
    :param c: right-hand multiple operand
    :return: result
    """
    if a.data.dtype != jnp.float32:
        raise ValueError("cpu_dense_ft_fma a array only supports float32 dtype")
  
    if b.data.dtype != jnp.float32:
        raise ValueError("cpu_dense_ft_fma b array only supports float32 dtype")

    if c.data.dtype != jnp.float32:
        raise ValueError("cpu_dense_ft_fma c array only supports float32 dtype")

    _check_basis_compat(a.basis, b.basis, c.basis)

    # FIXME review default basis, this is worked from ft_fma in roughpy/compute/__init__.py
    basis = b.basis
    out_depth = c.basis.depth
    lhs_depth = -1
    rhs_depth = -1

    call = jax.ffi.ffi_call(
        "cpu_dense_ft_fma",
        jax.ShapeDtypeStruct(a.data.shape, a.data.dtype)
    )

    return call(
        basis.degree_begin,
        a.data,
        b.data,
        c.data,
        width=basis.width,
        depth=basis.depth,
        out_depth=out_depth,
        lhs_depth=lhs_depth,
        rhs_depth=rhs_depth
    )

FreeTensor = DenseFreeTensor
