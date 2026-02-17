from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp
from jax import lax

from .concepts import Stream, BasisLike, LieT, GroupT
from ..intervals import Interval, Partition
from ..algebra import (
    FreeTensor,
    TensorBasis, 
    lie_to_tensor, 
    tensor_to_lie, 
    ft_fmexp, 
    ft_exp, 
    ft_log,
)


@dataclass(frozen=True)
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
        return Interval(
            inf=self._partition.inf,
            sup=self._partition.sup,
            interval_type=self._partition.interval_type,
        )
    
    def log_signature(self, interval: Interval) -> LieT:
        """Compute the log signature over an interval."""
        # This is a piecewise abelian stream, so the log signature over an 
        # interval is just the sum of the log signatures over the subintervals 
        # that intersect with the query interval.
        # 
        # This is a very simple implementation, and can be optimized by precomputing the
        # cumulative sums of the log signatures over the partition intervals.
        # 
        # associate scan of exps of the l2t of each piecewise segment, 
        # then take the log of the product of the exps over the query interval?
        initial = FreeTensor.identity(tensor_basis=self.group_basis)
        def compute_piece(acc, x_and_interval):
            x, p  = x_and_interval
            # NOTE: Might be a clearer way of writing this?
            # NOTE: the intersection length can be None?
            intersection = p.intersection(interval)
            if intersection is None:
                scale_factor = 1.0
                t = FreeTensor.identity(tensor_basis=self.group_basis)
            else:
                scale_factor = intersection.length / self._partition.length
                t = lie_to_tensor(x, tensor_basis=self.group_basis, scale_factor=scale_factor)
 
            return ft_fmexp(acc, t, tensor_basis=self.group_basis)
        
        intervals = self._partition.to_intervals()
        result = lax.associative_scan(compute_piece, initial, zip(self._data, intervals))
        return tensor_to_lie(result, lie_basis=self.lie_basis)

    
    # NOTE JL: Should this take a partition instead?
    def signature(self, interval: Interval) -> GroupT:
        """Compute the signature over an interval."""
        # exponentiate the log signature
        return ft_exp(self.log_signature(interval), self.group_basis)

    
def to_piecewise_abelian(
        stream: Stream[LieT, GroupT], partition: Partition                 
    ) -> PiecewiseAbelianStream[LieT, GroupT]:
    """Convert a stream to a piecewise abelian stream."""
    ...