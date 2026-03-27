from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax

from roughpy_jax.algebra import (
    FreeTensor,
    ft_fmexp,
    lie_to_tensor,
    to_log_signature,
    to_signature,
)
from roughpy_jax.bases import Basis
from roughpy_jax.intervals import Interval, Partition, RealInterval, intersection

from .concepts import GroupT, LieT, Stream


def _pas_dataclass(cls):
    """Helper to apply the dataclass and register_dataclass decorators in the correct order."""
    cls = dataclass(cls, frozen=True)
    return jax.tree_util.register_dataclass(
        cls,
        data_fields=["_data", "_partition"],
        meta_fields=["_lie_basis", "_group_basis"],
    )


@_pas_dataclass
class PiecewiseAbelianStream(Stream[LieT, GroupT]):
    """A stream representing a piecewise abelian path."""

    _data: tuple[LieT, ...]
    _partition: Partition
    _lie_basis: Basis
    _group_basis: Basis

    def __post_init__(self):
        """Validate the piecewise abelian stream."""
        if len(self._data) != len(self._partition):
            raise ValueError(
                f"Data length {len(self._data)} must match number "
                f"of intervals in partition {len(self._partition)}."
            )

    @property
    def lie_basis(self) -> Basis:
        """Return the Lie basis."""
        return self._lie_basis

    @property
    def group_basis(self) -> Basis:
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

    @property
    def dtype(self):
        """Return the coefficient dtype of the stream values."""
        return self._data[0].data.dtype

    @property
    def batch_dims(self) -> tuple[int, ...]:
        """Return the leading batch dimensions of the stream values."""
        return self._data[0].data.shape[:-1]

    @jax.jit
    def log_signature(self, interval: Interval) -> LieT:
        """Compute the log signature over an interval."""
        inf = jnp.asarray(interval.inf)
        sup = jnp.asarray(interval.sup)
        if inf.size != 1 or sup.size != 1:
            raise ValueError(
                "PiecewiseAbelianStream only supports scalar interval endpoints "
                "or single-element endpoint arrays"
            )
        if inf.shape or sup.shape:
            interval = RealInterval(inf.reshape(()), sup.reshape(()), interval.interval_type)

        initial = FreeTensor.identity(
            self._group_basis,
            dtype=self.dtype,
            batch_dims=self.batch_dims,
        )

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
            intersection_length = intersection(p, interval).length
            scale_factor = intersection_length / p.length
            return jax.lax.cond(
                intersection_length > 0,
                lambda: lie_to_tensor(x, scale_factor=scale_factor),
                lambda: FreeTensor.identity(
                    self._group_basis,
                    dtype=x.data.dtype,
                    batch_dims=self.batch_dims,
                ),
            )

        intervals = self._partition.to_intervals()
        all_tensors = [initial] + [
            get_piece((x, p)) for x, p in zip(self._data, intervals, strict=True)
        ]

        # Stack all tensors along a leading axis into a single batched FreeTensor.
        batched = jax.tree.map(lambda *arrs: jnp.stack(arrs), *all_tensors)

        result_batched = lax.associative_scan(
            lambda a, b: ft_fmexp(a, b, self._group_basis),
            batched,
        )

        # Take the last prefix (the full product over all selected pieces).
        result = jax.tree.map(lambda x: x[-1], result_batched)
        return to_log_signature(result)

    @jax.jit
    def signature(self, interval: Interval) -> GroupT:
        """Compute the signature over an interval."""
        log_sig = self.log_signature(interval)
        return to_signature(log_sig, tensor_basis=self._group_basis)


def to_piecewise_abelian(
    stream: Stream[LieT, GroupT], partition: Partition
) -> PiecewiseAbelianStream[LieT, GroupT]:
    """Convert a stream to a piecewise abelian stream."""
    ...
