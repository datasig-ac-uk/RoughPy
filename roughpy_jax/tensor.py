from dataclasses import dataclass
import jax
import jax.numpy as jnp

from .basis import TensorBasis, LieBasis


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
"""
FreeTensor = DenseFreeTensor
ShuffleTensor = DenseShuffleTensor
Lie = DenseLie
