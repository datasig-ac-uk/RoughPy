from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.lax as lax
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
            
            # FIXME JAX_ENABLE_X64 necessary to ensure final array is int64
            init = jnp.array(0, dtype=jnp.int64)
            _, degree_begin = lax.scan(degree_begin_fn, init, length=depth + 2)

        self.degree_begin = degree_begin

    def __eq__(self, rhs: "TensorBasis"):
        if self.width != rhs.width:
            return False
        if self.depth != rhs.depth:
            return False
        if self.degree_begin != rhs.degree_begin:
            return False
        return True


class DenseFreeTensor(NamedTuple):
    data: jnp.ndarray
    basis: TensorBasis


def dense_ft_fma(
    a: DenseFreeTensor,
    b: DenseFreeTensor,
    c: DenseFreeTensor
) -> DenseFreeTensor:
    # FIXME JAX_ENABLE_X64 separate method for 64?
    # if a.data.dtype != jnp.float64:
    #     raise ValueError("cpu_dense_ft_fma a array only supports float64 dtype")
  
    # if b.data.dtype != jnp.float64:
    #     raise ValueError("cpu_dense_ft_fma b array only supports float64 dtype")

    # if c.data.dtype != jnp.float64:
    #     raise ValueError("cpu_dense_ft_fma c array only supports float64 dtype")

    if a.data.dtype != jnp.float32:
        raise ValueError("cpu_dense_ft_fma a array only supports float32 dtype")
  
    if b.data.dtype != jnp.float32:
        raise ValueError("cpu_dense_ft_fma b array only supports float32 dtype")

    if c.data.dtype != jnp.float32:
        raise ValueError("cpu_dense_ft_fma c array only supports float32 dtype")

    call = jax.ffi.ffi_call(
        "cpu_dense_ft_fma",

        # FIXME which tensor drives result's shape?
        jax.ShapeDtypeStruct(a.data.shape, a.data.dtype)
    )

    # FIXME experimental passing all bases whilst getting linkage working. Change
    # to arguments of py_dense_ft_fma in roughpy/compute/_src/dense_basic.cpp
    # using one basis and max degrees.
    return call(
        a.basis.degree_begin,
        b.basis.degree_begin,
        c.basis.degree_begin,
        a.data,
        b.data,
        c.data,
        a_width=a.basis.width,
        a_depth=a.basis.depth,
        b_width=b.basis.width,
        b_depth=b.basis.depth,
        c_width=c.basis.width,
        c_depth=c.basis.depth,
    )


# Tensor aliases
FreeTensor = DenseFreeTensor
