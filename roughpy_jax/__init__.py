from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.lax as lax
from typing import NamedTuple

# try:
#     import _roughpy_jax_cpu
# except ImportError as e:
#     _roughpy_jax_cpu = None
#     raise ImportError("RoughPy JAX CPU backend is not installed correctly") from e
# else:
#     jax.ffi.register_ffi_target("cpu_dense_ft_fma", "cpu")


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
            _, degree_begin = lax.scan(degree_begin_fn, 0, length=depth + 2)

        self.degree_begin = degree_begin


class DenseFreeTensor(NamedTuple):
    data: jnp.ndarray
    basis: TensorBasis


# Tensor aliases
FreeTensor = DenseFreeTensor
