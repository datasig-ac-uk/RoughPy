from dataclasses import dataclass
from functools import partial
from typing import Tuple

import jax.numpy as jnp
from jax import lax
import jax

from .concepts import Stream, BasisLike, LieT, GroupT
from roughpy_jax.intervals import RealInterval, Partition, Interval
from roughpy_jax.algebra import (
    FreeTensor,
    lie_to_tensor, 
    tensor_to_lie, 
    ft_fmexp, 
    ft_exp, 
    ft_log,
)


def _pas_dataclass(cls):
    """Helper to apply the dataclass and register_dataclass decorators in the correct order."""
    cls = dataclass(cls, frozen=True)
    return jax.tree_util.register_dataclass(
        cls, data_fields=["_data", "_partition"],
        meta_fields=["_lie_basis", "_group_basis"],
    )


@_pas_dataclass
class PiecewiseAbelianStream(Stream[LieT, GroupT]):
    """A stream representing a piecewise abelian path."""
    
    _data: Tuple[LieT, ...]
    _partition: Partition
    _lie_basis: BasisLike
    _group_basis: BasisLike

    def __post_init__(self):
        """Validate the piecewise abelian stream."""
        if len(self._data) != len(self._partition):
            raise ValueError(f"Data length {len(self._data)} must match number "
                             f"of intervals in partition {len(self._partition)}."
                             )

    @property
    def lie_basis(self) -> BasisLike:
        """Return the Lie basis."""
        return self._lie_basis
    
    @property
    def group_basis(self) -> BasisLike:
        """Return the group basis."""
        return self._group_basis
    
    @property
    def support(self) -> Interval:
        """Return the support interval."""
        return RealInterval(
            _inf=self._partition.inf,
            _sup=self._partition.sup,
            _interval_type=self._partition.interval_type,
        )

    @jax.jit
    def log_signature(self, interval: Interval) -> LieT:
        """Compute the log signature over an interval."""
        # TODO: #303 replace this with the convenience function when it exists
        initial = self._get_identity(dtype=self._data[0].data.dtype)
        
        def get_piece(x_and_interval):
            """
            Get the tensor representation of a piece of the stream, scaled by the
            intersection length with the query interval. This is designed to be
            JIT-compilable.
            """
            # NOTE: This could be made more vectorized by processing all pieces at once.
            # UPDATE: Tried and it wasn't faster, and made the code more 
            # complicated, so leaving it as is for now.
            x, p = x_and_interval
            intersection_length = p.intersection(interval).length
            scale_factor = intersection_length / p.length
            return jax.lax.cond(
                intersection_length > 0,
                lambda: lie_to_tensor(x, tensor_basis=self._group_basis, scale_factor=scale_factor),
                # TODO: #303 replace this with the function on basis when it exists
                lambda: self._get_identity(dtype=x.data.dtype),
            )
        
        intervals = self._partition.to_intervals()
        all_tensors = [initial] + [get_piece((x, p)) for x, p in zip(self._data, intervals)]

        # Stack all tensors along a leading axis into a single batched FreeTensor.
        batched = jax.tree.map(lambda *arrs: jnp.stack(arrs), *all_tensors)

        result_batched = lax.associative_scan(
            lambda a, b: ft_fmexp(a, b, self._group_basis),
            batched,
        )

        # Take the last prefix (the full product over all selected pieces).
        result = jax.tree.map(lambda x: x[-1], result_batched)
        return tensor_to_lie(ft_log(result), lie_basis=self.lie_basis)

    def _get_identity(self, dtype) -> FreeTensor:
        """Return the identity element of the group."""
        identity_data = jnp.zeros((*self._data[0].data.shape[:-1], self._group_basis.size()), dtype=dtype).at[..., 0].set(1)
        return FreeTensor(identity_data, self._group_basis)

    @jax.jit
    def signature(self, interval: Interval) -> GroupT:
        """Compute the signature over an interval."""
        log_sig = self.log_signature(interval)
        tensor = lie_to_tensor(log_sig, tensor_basis=self._group_basis)
        return ft_exp(tensor, self._group_basis)


def to_piecewise_abelian(
        stream: Stream[LieT, GroupT], partition: Partition                 
    ) -> PiecewiseAbelianStream[LieT, GroupT]:
    """Convert a stream to a piecewise abelian stream."""
    ...