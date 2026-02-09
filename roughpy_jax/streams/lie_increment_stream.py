import itertools
import functools
import math

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

    def _query_cache(self, inf: float, sup: float) -> Lie:
        """
        Stub for dyadic cache lookup.

        This should return a JAX array shaped like (..., LieDim) containing
        the log-signature over [inf, sup] at the given resolution.
        """
        rounder = (
            math.floor if self._interval_type == IntervalType.ClOpen else math.ceil
        )

        diff = sup - inf
        _, exp = math.frexp(diff)
        exp = -exp
        begin = int(rounder(math.ldexp(inf, self._resolution)))
        end = int(rounder(math.ldexp(sup, self._resolution)))

        # When the query interval is smaller than the shortest dyadics provided here then special care is needed
        # We have to determine if the included end of any max-resolution interval is contained in the query interval.
        # This should be the case if the rounded inf and sup are different, in which case it is just a matter of
        # selecting the one that lies inside the interval. Which endpoint this is depends on the direction of rounding.
        if exp > self._resolution:
            if begin == end:
                return self._zero_log_signature()

            k = end if self._interval_type == IntervalType.ClOpen else begin
            return self._query_dyadic(k, self._resolution)

        shift = self._resolution - exp - 1
        pad = (1 << shift) - 1
        inf_trim = ((begin + pad) >> shift) << shift
        sup_trim = (end >> shift) << shift

        inf_work = inf_trim
        sup_work = sup_trim
        inf_indicator = inf_trim - inf
        sup_indicator = sup - sup_trim

        mid = math.ldexp(0.5, exp)

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

        result = self._query_cache(inf, sup)
        return result

    def signature(
        self, interval: Interval | None = None, resolution: int | None = None
    ) -> FreeTensor:
        log_sig = self.log_signature(interval)
        tensor = lie_to_tensor(log_sig, tensor_basis=self._group_basis)
        return ft_exp(tensor, out_basis=self._group_basis)
