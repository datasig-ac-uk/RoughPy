import math
from collections import deque

import pytest

from roughpy_jax.intervals import IntervalType, RealInterval
from roughpy_jax.streams.lie_increment_stream import dyadic_query


def _endpoint(k: int, n: int) -> float:
    return math.ldexp(float(k), -n)


def _contains(query: RealInterval, value: float) -> bool:
    if query.interval_type == IntervalType.ClOpen:
        return query.inf <= value < query.sup
    return query.inf < value <= query.sup


def _collect_endpoints(
    query: RealInterval,
    resolution: int,
    cache_interval_type: IntervalType,
) -> list[float]:
    def init(k1: int, k2: int, n: int) -> deque[float]:
        assert k1 <= k2
        assert (k2 - k1) <= 1

        if k1 == k2:
            return deque()

        # The only non-empty init case under the invariant is k2 == k1 + 1.
        k = k1 if cache_interval_type == IntervalType.ClOpen else k2
        return deque((_endpoint(k, n),))

    def get_left(k: int, n: int, digit: int) -> float | None:
        if not digit:
            return None
        return _endpoint(k, n)

    def get_right(k: int, n: int, digit: int) -> float | None:
        if not digit:
            return None
        return _endpoint(k, n)

    def combine(
        left: float | None, acc: deque[float], right: float | None
    ) -> deque[float]:
        if left is not None:
            acc.appendleft(left)
        if right is not None:
            acc.append(right)
        return acc

    return list(
        dyadic_query(
            query=query,
            resolution=resolution,
            init=init,
            get_left=get_left,
            get_right=get_right,
            combine=combine,
            cache_interval_type=cache_interval_type,
        )
    )


def _nudge_to_cache_type(
    query: RealInterval, cache_interval_type: IntervalType, resolution: int
) -> RealInterval:
    if query.interval_type == cache_interval_type:
        return RealInterval(query.inf, query.sup, cache_interval_type)

    if (
        query.interval_type == IntervalType.OpenCl
        and cache_interval_type == IntervalType.ClOpen
    ):
        inf_scaled = math.ldexp(query.inf, resolution)
        if inf_scaled == math.ceil(inf_scaled):
            inf = math.ldexp(int(inf_scaled) + 1, -resolution)
        else:
            inf = query.inf
        return RealInterval(inf, query.sup, cache_interval_type)

    if (
        query.interval_type == IntervalType.ClOpen
        and cache_interval_type == IntervalType.OpenCl
    ):
        sup_scaled = math.ldexp(query.sup, resolution)
        if sup_scaled == math.floor(sup_scaled):
            sup = math.ldexp(int(sup_scaled) - 1, -resolution)
        else:
            sup = query.sup
        return RealInterval(query.inf, sup, cache_interval_type)

    raise AssertionError("unsupported interval type conversion")


@pytest.mark.parametrize(
    "query,cache_interval_type",
    [
        (RealInterval(0.0, 1.0, IntervalType.ClOpen), IntervalType.ClOpen),
        (RealInterval(0.0, 1.0, IntervalType.OpenCl), IntervalType.OpenCl),
        (RealInterval(0.125, 0.875, IntervalType.ClOpen), IntervalType.ClOpen),
        (RealInterval(0.125, 0.875, IntervalType.OpenCl), IntervalType.OpenCl),
    ],
)
def test_deque_callbacks_return_ordered_counted_endpoints(query, cache_interval_type):
    result = _collect_endpoints(
        query, resolution=6, cache_interval_type=cache_interval_type
    )

    assert result == sorted(result)
    assert all(_contains(query, x) for x in result)


@pytest.mark.parametrize(
    "query,cache_interval_type,resolution",
    [
        (RealInterval(0.0, 1.0, IntervalType.OpenCl), IntervalType.ClOpen, 6),
        (RealInterval(0.0, 1.0, IntervalType.ClOpen), IntervalType.OpenCl, 6),
        (RealInterval(0.25, 0.75, IntervalType.OpenCl), IntervalType.ClOpen, 6),
        (RealInterval(0.25, 0.75, IntervalType.ClOpen), IntervalType.OpenCl, 6),
        (RealInterval(0.2, 0.8, IntervalType.OpenCl), IntervalType.ClOpen, 8),
        (RealInterval(0.2, 0.8, IntervalType.ClOpen), IntervalType.OpenCl, 8),
    ],
)
def test_mismatch_matches_inward_nudged_same_cache_type(
    query, cache_interval_type, resolution
):
    nudged = _nudge_to_cache_type(query, cache_interval_type, resolution)

    mismatch_result = _collect_endpoints(query, resolution, cache_interval_type)
    nudged_result = _collect_endpoints(nudged, resolution, cache_interval_type)

    assert mismatch_result == nudged_result


def test_opencl_query_on_clopen_cache_excludes_left_endpoint_if_dyadic():
    query = RealInterval(0.25, 0.75, IntervalType.OpenCl)
    result = _collect_endpoints(
        query, resolution=6, cache_interval_type=IntervalType.ClOpen
    )

    assert query.inf not in result


def test_clopen_query_on_opencl_cache_excludes_right_endpoint_if_dyadic():
    query = RealInterval(0.25, 0.75, IntervalType.ClOpen)
    result = _collect_endpoints(
        query, resolution=6, cache_interval_type=IntervalType.OpenCl
    )

    assert query.sup not in result


def test_init_returns_empty_if_k1_equals_k2():
    query = RealInterval(0.5, 0.5, IntervalType.ClOpen)
    result = _collect_endpoints(
        query,
        resolution=6,
        cache_interval_type=IntervalType.ClOpen,
    )
    assert result == []


@pytest.mark.parametrize(
    "query,cache_interval_type,resolution,expected",
    [
        # ClOpen cache: included endpoints are left endpoints k/2^n.
        (RealInterval(0.100, 0.120, IntervalType.ClOpen), IntervalType.ClOpen, 3, []),
        (
            RealInterval(0.125, 0.200, IntervalType.ClOpen),
            IntervalType.ClOpen,
            3,
            [0.125],
        ),
        (RealInterval(0.100, 0.125, IntervalType.OpenCl), IntervalType.ClOpen, 3, []),
        # OpenCl cache: included endpoints are right endpoints k/2^n.
        (RealInterval(0.780, 0.790, IntervalType.OpenCl), IntervalType.OpenCl, 3, []),
        (
            RealInterval(0.700, 0.750, IntervalType.OpenCl),
            IntervalType.OpenCl,
            3,
            [0.75],
        ),
        (RealInterval(0.750, 0.790, IntervalType.ClOpen), IntervalType.OpenCl, 3, []),
    ],
)
def test_short_intervals_include_resolution_endpoint_if_and_only_if_contained(
    query,
    cache_interval_type,
    resolution,
    expected,
):
    result = _collect_endpoints(
        query=query,
        resolution=resolution,
        cache_interval_type=cache_interval_type,
    )
    assert result == expected


@pytest.mark.parametrize(
    "query,cache_interval_type",
    [
        (RealInterval(0.25, 0.25, IntervalType.ClOpen), IntervalType.ClOpen),
        (RealInterval(0.25, 0.25, IntervalType.OpenCl), IntervalType.OpenCl),
        (RealInterval(0.25, 0.25, IntervalType.ClOpen), IntervalType.OpenCl),
        (RealInterval(0.25, 0.25, IntervalType.OpenCl), IntervalType.ClOpen),
    ],
)
def test_empty_intervals_always_return_empty(query, cache_interval_type):
    result = _collect_endpoints(
        query=query,
        resolution=6,
        cache_interval_type=cache_interval_type,
    )
    assert result == []
