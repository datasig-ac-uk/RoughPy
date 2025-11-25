from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp

# FIXME review point: neatly load compute module
import sys
sys.path.append('roughpy/compute')
import _rpy_compute_internals


# For exposition only
# class TensorBasis:
#     width: np.int32
#     depth: np.int32
#     degree_begin: np.ndarray[tuple[typing.Any], np.intp]
@partial(jax.tree_util.register_dataclass, data_fields=[], meta_fields=[])
class TensorBasis(_rpy_compute_internals.TensorBasis):
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


"""
Tensor aliases. Tensors are assumed to be dense without prefix
"""
FreeTensor = DenseFreeTensor
ShuffleTensor = DenseShuffleTensor
