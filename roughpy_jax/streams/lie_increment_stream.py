import itertools

from typing import Type, TypeVar, Optional

import numpy as np
import jax
import jax.numpy as jnp

from jax.typing import ArrayLike

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


def _batching_loop(shape):
    if not shape:
        yield ()
        return

    ranges = tuple(map(range, shape))
    yield from itertools.product(*ranges)


def _unpack_lie_to_array_tuple(lies: list[Lie]) -> tuple[jax.Array, ...]:
    arrays = [[]]

    for lie in lies:
        arrays[0].append(lie.data)

    return tuple(jnp.vstack(arr) for arr in arrays)


def _extend_cache_from_base(
    base: list[Lie], resolution, cache_basis
) -> tuple[jax.Array, ...]:
    tensor_basis = TensorBasis(cache_basis.width, cache_basis.depth)

    def g(r, previous):
        for k in range(1 << r):
            k1 = 2 * k
            k2 = k1 + 1
            tensor = ft_exp(lie_to_tensor(previous[k1]), out_basis=tensor_basis)
            tensor = ft_fmexp(
                tensor, lie_to_tensor(previous[k2]), out_basis=tensor_basis
            )
            lie = tensor_to_lie(ft_log(tensor), lie_basis=cache_basis)
            yield lie

    prev_size = 0
    for res in range(resolution - 1, -1, -1):
        next_size = len(base)
        base.extend(g(res, base[prev_size:]))
        prev_size = next_size

    return _unpack_lie_to_array_tuple(base)


def _ft_identity(
    basis: TensorBasis, batch_dims: tuple[int, ...], dtype: jnp.dtype
) -> FreeTensor:
    data = jnp.zeros((*batch_dims, basis.size()), dtype=dtype)
    data[..., 0] = 1

    return FreeTensor(data, basis)


def _data_to_dyadic_cache(
    timestamps: ArrayLike,
    data: ArrayLike,
    resolution: int,
    input_basis: LieBasis,
    cache_basis: LieBasis,
    interval_type=IntervalType.ClOpen,
    integer_type=jnp.int32,
) -> tuple[jax.Array, ...]:
    tensor_basis = TensorBasis(width=cache_basis.width, depth=cache_basis.depth)
    base_components = [[] for _ in range(1 << resolution)]

    *shape, data_dim = data.shape
    pad_size = input_basis.size() - data_dim
    padded_data = jnp.pad(data, [(0, 0)] * (len(shape)) + [(0, pad_size)])

    rounder = jnp.floor if interval_type == IntervalType.ClOpen else jnp.ceil
    k_array = rounder(jnp.ldexp(timestamps, resolution)).astype(integer_type)

    change_points = jnp.nonzero(k_array[1:] != k_array[:-1])

    last_change_idx = 0
    for idx in change_points:
        tensor_inc = jax.lax.reduce(
            [
                Lie(padded_data[..., i, :], input_basis)
                for i in range(last_change_idx, idx)
            ],
            _ft_identity(tensor_basis, shape, data.dtype),
            lambda acc, x: ft_fmexp(acc, lie_to_tensor(x), out_basis=tensor_basis),
            dimensions=[0],
        )

        k = k_array[last_change_idx]
        base_components[k].append(tensor_to_lie(ft_log(tensor_inc)))

    return _extend_cache_from_base(base, resolution, cache_basis)


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

        return _extend_cache_from_base(
            lies, resolution, stream.lie_basis, stream.lie_basis
        )

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

    @classmethod
    def from_increments(
        cls: Type[T],
        timestamps: ArrayLike,
        data: ArrayLike,
        *,
        resolution: Optional[int],
        input_data_basis: Optional[LieBasis],
        lie_basis: Optional[LieBasis],
        interval_type=IntervalType.ClOpen,
        **kwargs,
    ) -> T:
        data_array = jnp.asarray(data)
        ts_array = jnp.asarray(timestamps)

        if data_array.ndim < 2:
            raise ValueError("data must be at least 2D")

        if ts_array.ndim != 1:
            raise ValueError("timestamps must be 1D")

        *batch_dims, time_dim, data_dim = data_array.shape
        if time_dim != ts_array.shape[-1]:
            raise ValueError(
                "data and timestamps must have the same number of time points"
            )

        # First sort out the algebra details
        if input_data_basis is None:
            input_data_basis = LieBasis(width=data_dim, depth=1)
        elif data_dim != input_data_basis.size():
            raise ValueError("data dimension should match input_data_basis size")

        if lie_basis is None:
            width = kwargs.pop("width", data_dim)
            depth = kwargs.pop("depth", data_dim)
            lie_basis = LieBasis(width=width, depth=depth)

        # Now sort out the support and scale the data
        inf = jnp.min(ts_array)
        sup = jnp.max(ts_array)

        if interval_type == IntervalType.ClOpen:
            sup = jnp.nextafter(sup, jnp.inf)
        elif interval_type == IntervalType.OpenCl:
            inf = jnp.nextafter(inf, -jnp.inf)

        support = RealInterval(inf, sup, interval_type)

        # Adjust the timestamps so they lie in the unit interval
        sf = ts_array.dtype.type(sup - inf)
        shift = ts_array.dtype.type(inf)
        ts_array = sf * ts_array - shift

        if resolution is None:
            time_increments = jnp.diff(ts_array, axis=-1)
            min_diff = jnp.min(time_increments)

            _, exp = jnp.frexp(min_diff)
            resolution = int(1 - exp)

        cache = _data_to_dyadic_cache(
            ts_array, data_array, resolution, input_data_basis, lie_basis
        )

        return cls(cache, lie_basis, support, **kwargs)

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
