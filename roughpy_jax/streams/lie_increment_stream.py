from typing import Type, TypeVar

import jax.numpy as jnp

from roughpy_jax.algebra import (
    Lie,
    LieBasis,
    TensorBasis,
    FreeTensor,
    lie_to_tensor,
    ft_exp,
    ft_fmexp,
    ft_log,
    tensor_to_lie,
)
from roughpy_jax.intervals import Interval, IntervalType, RealInterval, DyadicInterval

from .concepts import Stream

T = TypeVar("T")


# TODO: replace this with a standardised version imported from intervals
def intersection(ivl1: Interval, ivl2: Interval) -> RealInterval:
    if ivl1.interval_type != ivl2.interval_type:
        raise ValueError(
            "intersection between intervals of different types is not supported"
        )

    return RealInterval(
        max(ivl1.inf, ivl2.inf),
        min(ivl1.sup, ivl2.sup),
        ivl1.interval_type,
    )


class LieIncrementStream(Stream[Lie, FreeTensor]):
    """
    Stream backed by a contiguous cache of dyadic log-signatures.

    The cache is a JAX array with shape (..., 2^(R+1), LieDim), where the
    cache axis packs log-signatures over dyadic intervals of lengths between
    2^-R and 1 in steps of 2. The final element of the cache axis is unused
    by the geometric series of dyadic intervals and should be zero.
    """

    @staticmethod
    def _cache_length_from_resolution(resolution: int) -> int:
        return 1 << (int(resolution) + 1)

    def __init__(
        self,
        cache: jnp.ndarray,
        lie_basis: LieBasis,
        support: Interval | None = None,
        group_basis: TensorBasis | None = None,
        resolution: int | None = None,
    ) -> None:
        if cache.ndim < 2:
            raise ValueError("cache must have shape (..., cache_length, lie_dim)")

        lie_dim = int(lie_basis.size())
        if cache.shape[-1] != lie_dim:
            raise ValueError(
                f"cache lie dimension mismatch: expected {lie_dim}, got {cache.shape[-1]}"
            )

        cache_length = int(cache.shape[-2])
        expected_length = self._cache_length_from_resolution(resolution)
        if cache_length != expected_length:
            raise ValueError(
                f"cache length mismatch for resolution {resolution}: "
                f"expected {expected_length}, got {cache_length}"
            )

        self._cache = cache
        self._lie_basis = lie_basis
        self._group_basis = group_basis or TensorBasis(lie_basis.width, lie_basis.depth)
        self._support = support or RealInterval(0.0, 1.0, IntervalType.ClOpen)
        self._resolution = int(resolution)

    @staticmethod
    def _stream_to_cache(
        stream: Stream,
        resolution: int,
        interval_type: IntervalType = IntervalType.ClOpen,
    ) -> jnp.ndarray:

        inf = stream.support.inf
        scale_factor = stream.support.sup - inf

        def reparam(di):
            return RealInterval(
                interval_type, di.inf * scale_factor + inf, di.sup * scale_factor + inf
            )

        def f(k, r):
            di = DyadicInterval(k, r, interval_type)
            query = reparam(di)
            log_sig = stream.log_signature(query)
            return log_sig.data

        lies = [f(k, resolution) for k in range(1 << resolution)]

        def g(r, previous):
            for k in range(1 << r):
                k1 = 2 * k
                k2 = k1 + 1
                tensor = ft_exp(lie_to_tensor(previous[k1]))
                tensor = ft_fmexp(tensor, lie_to_tensor(previous[k2]))
                lie = tensor_to_lie(ft_log(tensor))
                yield lie.data

        prev_size = 0
        for res in range(resolution - 1, -1, -1):
            next_size = len(lies)
            lies.extend(g(res, lies[prev_size:]))
            prev_size = next_size

        return jnp.vstack(lies)

    @classmethod
    def from_stream(cls: Type[T], stream: Stream, resolution: int) -> T:
        lie_basis = stream.lie_basis
        group_basis = stream.group_basis
        support = stream.support

        if resolution <= 0:
            raise ValueError(f"resolution must be positive, got {resolution}")

        if (fun := getattr(stream, "__dyadic_cache__")) is not None:
            cache = jnp.asarray(fun(resolution))
        else:
            cache = cls._stream_to_cache()

        new_stream = cls(cache, lie_basis, group_basis, support, resolution)

        new_stream.__base_stream__ = stream

        return new_stream

    @property
    def lie_basis(self) -> LieBasis:
        return self._lie_basis

    @property
    def group_basis(self) -> TensorBasis:
        return self._group_basis

    @property
    def support(self) -> Interval:
        return self._support

    @property
    def resolution(self) -> int:
        return self._resolution

    def _query_cache(self, inf: float, sup: float) -> jnp.ndarray:
        """
        Stub for dyadic cache lookup.

        This should return a JAX array shaped like (..., LieDim) containing
        the log-signature over [inf, sup] at the given resolution.
        """
        raise NotImplementedError("dyadic cache querying is not implemented yet")

    def _zero_log_signature(self) -> Lie:
        batch_shape = self._cache.shape[:-2]
        lie_dim = int(self._lie_basis.size())
        zeros = jnp.zeros((*batch_shape, lie_dim), dtype=self._cache.dtype)
        return Lie(zeros, self._lie_basis)

    def _reparamterise(self, interval: Interval) -> tuple[float, float]:
        inf = self.support.inf
        length = self.support.sup - self.support.inf

        return (interval.inf - inf) / length, (interval.sup - inf) / length

    def log_signature(self, interval: Interval | None = None) -> Lie:
        if interval is None:
            interval = self._support

        query_interval = intersection(interval, self.support)
        if query_interval.sup <= query_interval.inf:
            return self._zero_log_signature()

        inf, sup = self._reparamterise(query_interval)

        data = self._query_cache(inf, sup)
        return Lie(data, self._lie_basis)

    def signature(
        self, interval: Interval | None = None, resolution: int | None = None
    ) -> FreeTensor:
        log_sig = self.log_signature(interval)
        tensor = lie_to_tensor(log_sig, tensor_basis=self._group_basis)
        return ft_exp(tensor, out_basis=self._group_basis)
