import itertools
import functools
import typing
import math

from typing import Type, TypeVar, Optional, Callable, TypeAlias, Any, Union

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


def _zero_lie(basis: LieBasis, batch_dims: tuple[int, ...], dtype: jnp.dtype) -> Lie:
    data = jnp.zeros((*batch_dims, basis.size()), dtype=dtype)
    return Lie(data, basis)


def _extend_cache_from_base(base: list[Lie], resolution, cache_basis) -> jax.Array:
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

    batch_dims = base[0].data.shape[:-1]
    dtype = base[0].data.dtype
    zero = _zero_lie(cache_basis, batch_dims, dtype)
    base.append(zero)

    cache = jnp.stack([lie.data for lie in base], axis=0)
    return cache


def _ft_identity(
    basis: TensorBasis, batch_dims: tuple[int, ...], dtype: jnp.dtype
) -> FreeTensor:
    data = jnp.zeros((*batch_dims, basis.size()), dtype=dtype)
    data[..., 0] = 1

    return FreeTensor(data, basis)


def _flatten(lies: list[Lie]) -> Lie:
    data = jnp.array([l.data for l in lies])
    return Lie(data, lies[0].basis)


def _cbh(
    data: jax.Array,
    data_basis: LieBasis,
    cache_basis: LieBasis,
    tensor_basis: TensorBasis | None = None,
    axis: int = 0,
) -> Lie:
    tensor_basis = tensor_basis or TensorBasis(
        width=cache_basis.width, depth=cache_basis.depth
    )

    batch_shape = data.shape[:axis] + data.shape[axis + 1 :]
    acc = _ft_identity(tensor_basis, batch_shape, data.dtype)

    for k in range(data.shape[axis]):
        lie = Lie(jnp.take(data, k, axis=axis), data_basis)
        ft_fmexp(acc, lie_to_tensor(lie), out_basis=tensor_basis)

    result = tensor_to_lie(ft_log(acc), lie_basis=cache_basis)
    return result


def _build_base_entry(
    k: int,
    k_arrays: list[jax.Array],
    data: list[jax.Array],
    data_basis: LieBasis,
    cache_basis: LieBasis,
    tensor_basis: TensorBasis,
) -> Lie:

    def inner(ks, ds):
        mask = ks == k
        if not jnp.any(mask):
            return _zero_lie(cache_basis, ds.shape[:-1], ds.dtype)

        return _cbh(ds[mask, ...], data_basis, cache_basis, tensor_basis)

    components = list(map(inner, k_arrays, data))
    return _flatten(components)


LeftT = TypeVar("LeftT")
RightT = TypeVar("RightT")
AccT = TypeVar("AccT")

DQInitT: TypeAlias = Callable[
    [int, int, int],
    AccT,
]
DQLeftGetterT: TypeAlias = Callable[[int, int, int], LeftT]
DQRightGetterT: TypeAlias = Callable[[int, int, int], RightT]
DQCombineT: TypeAlias = Callable[[LeftT, AccT, RightT], AccT]


def _resolve_short_case(
    inf_trim: int,
    sup_trim: int,
    inf_scaled: float,
    sup_scaled: float,
    is_clopen: bool,
) -> tuple[int, int]:
    if sup_trim < inf_trim:
        return inf_trim, inf_trim

    if is_clopen:
        k1 = inf_trim
        k2 = k1 + (0 if sup_trim == sup_scaled else 1)
        return k1, k2

    k2 = sup_trim
    k1 = k2 - (0 if inf_trim == inf_scaled else 1)

    return k1, k2


def dyadic_query(
    query: Interval,
    resolution: int,
    init: DQInitT,
    get_left: DQLeftGetterT,
    get_right: DQRightGetterT,
    combine: DQCombineT,
    cache_interval_type: IntervalType = IntervalType.ClOpen,
) -> AccT:
    is_clopen = cache_interval_type == IntervalType.ClOpen

    inf_scaled = math.ldexp(query.inf, resolution)
    sup_scaled = math.ldexp(query.sup, resolution)
    inf = math.ceil(inf_scaled)
    sup = math.floor(sup_scaled)

    # If query and cache interval types differ, nudge the endpoint that is
    # excluded by the query but included by the cache inward at dyadic points.
    if query.interval_type != cache_interval_type:
        if query.interval_type == IntervalType.OpenCl and is_clopen:
            if inf_scaled == inf:
                inf += 1
                inf_scaled = inf
        elif query.interval_type == IntervalType.ClOpen and not is_clopen:
            if sup_scaled == sup:
                sup -= 1
                sup_scaled = sup

    if sup < inf:
        return init(inf, inf, resolution)

    effective_width = math.ldexp(sup_scaled - inf_scaled, -resolution)
    _, exponent = math.frexp(effective_width)
    coarse_resolution = 1 - exponent

    # When the query interval is smaller than the shortest dyadics provided here then special care is needed
    # We have to determine if the included end of any max-resolution interval is contained in the query interval.
    # This should be the case if the rounded inf and sup are different, in which case it is just a matter of
    # selecting the one that lies inside the interval. Which endpoint this is depends on the direction of rounding.
    if coarse_resolution > resolution:
        k1, k2 = _resolve_short_case(inf, sup, inf_scaled, sup_scaled, is_clopen)
        return init(k1, k2, resolution)

    steps = resolution - coarse_resolution
    inf_working = (inf + ((1 << steps) - 1)) >> steps
    sup_working = sup >> steps

    if sup_working < inf_working:
        return init(inf_working, inf_working, coarse_resolution)

    result = init(inf_working, sup_working, coarse_resolution)

    for i in range(1, steps + 1):
        j = steps - i
        r = coarse_resolution + i

        inf_bit = (inf >> j) & 1
        sup_bit = (sup >> j) & 1

        # The update of working values needs to be handled with care. This depends on whether
        # the dyadic intervals are open on the right or left. In the first case (clopen) the
        # value passed as the first argument to get_right should be 2*sup_working and the
        # new bit should be added after this call. For get_left, the first argument should
        # be the fully updated 2*inf_working - inf_bit. In the second case, the reverse
        # holds: inf_working has a two-stage update and sup_working has a one-stage update.
        if is_clopen:
            left_k = (inf_working << 1) - inf_bit
            right_k = sup_working << 1
            left = get_left(left_k, r, inf_bit)
            right = get_right(right_k, r, sup_bit)
            inf_working = left_k
            sup_working = right_k + sup_bit
        else:
            left_k = inf_working << 1
            right_k = (sup_working << 1) + sup_bit
            left = get_left(left_k, r, inf_bit)
            right = get_right(right_k, r, sup_bit)
            inf_working = left_k - inf_bit
            sup_working = right_k

        result = combine(left, result, right)

    return result


class LieIncrementStream(Stream[Lie, FreeTensor]):
    """
    Stream backed by a contiguous cache of dyadic log-signatures.

    The cache is a JAX array with shape (2^(R+1), ..., LieDim), where the
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
        interval_type: IntervalType = IntervalType.ClOpen,
    ):
        if cache.ndim < 2:
            raise ValueError("cache must have shape (cache_length, ..., lie_dim)")

        lie_dim = int(lie_basis.size())
        if cache.shape[-1] != lie_dim:
            raise ValueError(
                f"cache lie dimension mismatch: expected {lie_dim}, got {cache.shape[-1]}"
            )

        cache_length = int(cache.shape[0])
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
        self._interval_type = interval_type
        self._zero_index = cache_length - 1

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
            return stream.log_signature(query)

        lies = [f(k, resolution) for k in range(1 << resolution)]

        return _extend_cache_from_base(lies, resolution, stream.lie_basis)

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
        timestamps: ArrayLike | list[ArrayLike],
        data: ArrayLike | list[ArrayLike],
        *,
        resolution: Optional[int],
        input_data_basis: Optional[LieBasis],
        lie_basis: Optional[LieBasis],
        interval_type=IntervalType.ClOpen,
        data_dtype: Optional[jnp.dtype] = None,
        time_dtype: jnp.dtype = jnp.dtype("float32"),
        dyadic_integer_type: jnp.dtype = jnp.dtype("int32"),
        **kwargs,
    ) -> T:

        if isinstance(timestamps, list):
            time_arrays = [jnp.asarray(ts) for ts in timestamps]
        else:
            time_arrays = [jnp.asarray(timestamps)]

        if isinstance(data, list):
            data_arrays = [jnp.asarray(ds) for ds in data]
        else:
            data_arrays = [jnp.asarray(data)]

        if not time_arrays or not data_arrays:
            raise ValueError("timestamps and data cannot be empty")

        if not len(time_arrays) == len(data_arrays):
            raise ValueError("timestamps and data must be the same length")

        time_lens = []
        mins = []
        maxs = []
        for ts in time_arrays:
            if ts.ndim != 1:
                raise ValueError("timestamps must be held in 1D arrays")

            time_lens.append(ts.shape[0])
            mins.append(jnp.min(ts))
            maxs.append(jnp.max(ts))

        ds = data_arrays[0]
        if ds.ndim != 2:
            raise ValueError("data arrays must be at least 2D")

        dt_dim, *batch_dims, lie_dim = ds.shape
        if dt_dim != time_lens[0]:
            raise ValueError(
                f"Time dimension mismatch at index 0: expected {time_lens[0]}, got {dt_dim}"
            )

        dtypes = [ds.dtype]
        for i, (ds, expected_dt) in enumerate(
            zip(data_arrays[1:], time_lens[1:]), start=1
        ):
            if ds.ndim < 2:
                raise ValueError("data arrays must be at least 2D")

            dt_dim, b_dims, *l_dim = ds.shape
            if dt_dim != expected_dt:
                raise ValueError(
                    f"Time dimension mismatch at index {i}: expected {expected_dt}, got {dt_dim}"
                )

            if b_dims != batch_dims:
                raise ValueError(
                    f"Batch dimension mismatch at index {i}: expected {batch_dims}, got {b_dims}"
                )

            dtypes.append(ds.dtype)

            if input_data_basis is not None:
                basis_size = input_data_basis.size()
                if l_dim > basis_size:
                    raise ValueError(
                        f"data dimension {l_dim} is incompatible with the specified data basis with size {basis_size}"
                    )
            elif l_dim != lie_dim:
                raise ValueError(
                    f"unable to determine appropriate data basis: inconsistent data dimensions at index {i}"
                )

        dtype = data_dtype or jnp.result_type(*dtypes)
        input_data_basis = input_data_basis or LieBasis(width=lie_dim, depth=1)
        basis_size = input_data_basis.size()

        for i in range(len(data_arrays)):
            *shape, l_dim = data_arrays[i].shape
            padding = [(0, 0)] * len(shape) + [(0, basis_size - l_dim)]
            data_arrays[i] = jnp.pad(data_arrays[i].astype(dtype), padding)

        if lie_basis is None:
            lie_basis = LieBasis(width=input_data_basis, depth=2)

        # Now sort out the support and scale the data
        if interval_type == IntervalType.ClOpen:
            sup = jnp.nextafter(max(*maxs), jnp.inf)
            inf = min(*mins)
        else:  # interval_type == IntervalType.OpenCl:
            sup = max(*maxs)
            inf = jnp.nextafter(min(*mins), -jnp.inf)

        support = RealInterval(inf, sup, interval_type)

        # Adjust the timestamps so they lie in the unit interval
        sf = time_dtype.type(sup - inf)
        shift = time_dtype.type(inf)
        time_arrays = [sf * ts.astype(time_dtype) - shift for ts in time_arrays]

        if resolution is None:
            min_diff = min(*(jnp.min(jnp.diff(ts, axis=-1)) for ts in time_arrays))
            _, exp = jnp.frexp(min_diff)
            resolution = int(1 - exp)

        tensor_basis = TensorBasis(width=lie_basis.width, depth=lie_basis.depth)

        rounder = jnp.floor if interval_type == IntervalType.ClOpen else jnp.ceil
        k_arrays = [
            rounder(jnp.ldexp(ts, resolution)).astype(dyadic_integer_type)
            for ts in timestamps
        ]

        f = functools.partial(
            _build_base_entry,
            k_arrays=k_arrays,
            data=data,
            data_basis=input_data_basis,
            cache_basis=lie_basis,
            tensor_basis=tensor_basis,
        )

        base = [f(k) for k in range(1 << resolution)]
        cache = _extend_cache_from_base(base, resolution, lie_basis)

        return cls(
            cache,
            lie_basis,
            support,
            resolution=resolution,
            group_basis=tensor_basis,
            **kwargs,
        )

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

    def _zero_log_signature(self) -> Lie:
        return Lie(self._cache[-1, ...], self._lie_basis)

    def _query_dyadic(self, k: int, n: int) -> Lie: ...

    def _query_cache(self, query: RealInterval) -> Lie:
        """
        Stub for dyadic cache lookup.

        This should return a JAX array shaped like (..., LieDim) containing
        the log-signature over [inf, sup] at the given resolution.
        """

    def _reparamterise(self, interval: Interval) -> RealInterval:
        inf = self.support.inf
        length = self.support.sup - self.support.inf

        return RealInterval(
            (interval.inf - inf) / length,
            (interval.sup - inf) / length,
            interval.interval_type,
        )

    def _query_init(self, k1: int, k2: int, n: int) -> Lie:
        if k1 == k2:
            return self._zero_log_signature()

        if self._interval_type == IntervalType.ClOpen:
            return self._query_dyadic(k1, n)

        return self._query_dyadic(k2, n)

    def _query_get(self, k: int, n: int, digit: int) -> Lie:
        if not digit:
            return self._zero_log_signature()

        return self._query_dyadic(k, n)

    def _query_combine(self, left: Lie, acc: Lie, right: Lie) -> Lie:
        ft_result = _ft_identity(
            self._group_basis, self.cache.shape[1:-1], acc.data.dtype
        )

        ft_result = ft_fmexp(ft_result, lie_to_tensor(left))
        ft_result = ft_fmexp(ft_result, lie_to_tensor(acc))
        ft_result = ft_fmexp(ft_result, lie_to_tensor(right))

        return tensor_to_lie(ft_log(ft_result))

    def log_signature(self, interval: Interval | None = None) -> Lie:
        if interval is None:
            interval = self._support

        query_interval = intersection(interval, self.support)
        if query_interval.sup <= query_interval.inf:
            return self._zero_log_signature()

        reparam_query = self._reparamterise(query_interval)

        result = dyadic_query(
            reparam_query,
            self._resolution,
            self._query_init,
            self._query_get,
            self._query_get,
            self._query_combine,
            jnp.dtype("int32"),
            self._interval_type,
        )

        return result

    def signature(
        self, interval: Interval | None = None, resolution: int | None = None
    ) -> FreeTensor:
        log_sig = self.log_signature(interval)
        tensor = lie_to_tensor(log_sig, tensor_basis=self._group_basis)
        return ft_exp(tensor, out_basis=self._group_basis)
