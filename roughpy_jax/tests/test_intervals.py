import dataclasses

import pytest
from roughpy_jax.intervals import (
    Dyadic,
    DyadicInterval,
    Interval,
    IntervalType,
    Partition,
    RealInterval,
    intersection,
)


@pytest.mark.parametrize(
    "interval_type",
    (IntervalType.ClOpen, IntervalType.OpenCl)
)
def test_protocol_conformance_runtime_checkable(interval_type):
    # All three concrete types should be instances of the Interval protocol
    di = DyadicInterval(k=2, n=1, _interval_type=interval_type)
    ri = RealInterval(_inf=1.0, _sup=2.0, _interval_type=interval_type)
    p = Partition(_endpoints=[0.0, 1.0], _interval_type=interval_type)

    assert isinstance(di, Interval)
    assert isinstance(ri, Interval)
    assert isinstance(p, Interval)


class TestDyadic:
    def test_dyadic_float_conversion_default(self):
        d1 = Dyadic(k=3, n=1)
        assert float(d1) == 1.5

        d2 = Dyadic(k=-1, n=2)
        assert float(d2) == -0.25

        d3 = Dyadic(k=5, n=0)
        assert float(d3) == 5.0
        
    def test_dyadic_float_conversion_with_float_inputs(self):
        d1 = Dyadic(k=3.0, n=1.0)
        assert float(d1) == 1.5

        d2 = Dyadic(k=-1.0, n=2.0)
        assert float(d2) == -0.25

        d3 = Dyadic(k=5.0, n=0.0)
        assert float(d3) == 5.0
        
    def test_dyadic_float_conversion_with_mixed_inputs(self):
        d1 = Dyadic(k=3, n=2.0)
        assert float(d1) == 0.75

        d2 = Dyadic(k=-1.0, n=3)
        assert float(d2) == -0.125

        d3 = Dyadic(k=5, n=2.0)
        assert float(d3) == 1.25
        
    def test_dyadic_negative_n(self):
        with pytest.raises(ValueError):
            Dyadic(k=1, n=-1)

class TestDyadicInterval:
    def test_dyadic_interval_clopen(self):
        di = DyadicInterval(k=3, n=1, _interval_type=IntervalType.ClOpen)
        assert di.inf == pytest.approx(1.5)
        assert di.sup == pytest.approx(2.0)
        assert di.length == pytest.approx(0.5)
        
    def test_dyadic_interval_opencl(self):
        di = DyadicInterval(k=3, n=1, _interval_type=IntervalType.OpenCl)
        assert di.inf == pytest.approx(1.0)
        assert di.sup == pytest.approx(1.5)
        assert di.length == pytest.approx(0.5)


class TestRealInterval:
    def test_real_interval_basic_and_frozen(self):
        ri = RealInterval(_inf=0.1, _sup=0.2, _interval_type=IntervalType.ClOpen)
        assert ri.interval_type is IntervalType.ClOpen
        assert ri.inf == pytest.approx(0.1)
        assert ri.sup == pytest.approx(0.2)

        # Frozen dataclass should not allow mutation
        with pytest.raises(dataclasses.FrozenInstanceError):
            ri._inf = 0.0
            
    def test_real_interval_length(self):
        ri = RealInterval(_inf=0.1, _sup=0.2, _interval_type=IntervalType.ClOpen)
        assert ri.length == pytest.approx(0.1)
        
    def test_real_interval_length_zero(self):
        ri = RealInterval(_inf=0.1, _sup=0.1, _interval_type=IntervalType.ClOpen)
        assert ri.length == pytest.approx(0.0)
    
    def test_real_interval_length_negative(self):
        # This is the principal edge case - the interval is invalid and length 
        # should be zero as that's how we determine no-overlap in intersection. 
        ri = RealInterval(_inf=0.2, _sup=0.1, _interval_type=IntervalType.ClOpen)
        assert ri.length == pytest.approx(0.0)
    
    def test_to_string(self):
        ri = RealInterval(_inf=0.1, _sup=0.2, _interval_type=IntervalType.ClOpen)
        assert str(ri) == "[0.1, 0.2)"
        ri2 = RealInterval(_inf=0.1, _sup=0.2, _interval_type=IntervalType.OpenCl)
        assert str(ri2) == "(0.1, 0.2]"


def test_partition_with_dyadic_endpoints_uses_float_conversion():
    # NOTE: Note sure this is actually a use-case we want to support, but it is 
    # currently supported and should be tested.
    endpoints = [Dyadic(k=0, n=0), Dyadic(k=1, n=0)]
    p = Partition(_endpoints=endpoints, _interval_type=IntervalType.ClOpen)
    assert p.inf == pytest.approx(0.0)
    assert p.sup == pytest.approx(1.0)
    

@pytest.fixture(params=[
    pytest.param(IntervalType.ClOpen, id="ClOpen"),
    pytest.param(IntervalType.OpenCl, id="OpenCl")
])
def interval_type(request) -> IntervalType:
    return request.param


@pytest.fixture()
def real_intervals(interval_type) -> tuple[RealInterval, RealInterval]:
    ri1 = RealInterval(_inf=0.0, _sup=1.0, _interval_type=interval_type)
    ri2 = RealInterval(_inf=0.5, _sup=1.5, _interval_type=interval_type)
    return ri1, ri2


class TestIntersection:
    """
    Tests for the intersection function, which computes the intersection of two 
    intervals. This has been pruned back to only work on RealInterval, as it's 
    the only type of interval we currently support intersection for.
    """
    
    @pytest.mark.parametrize(
        "left_interval,right_interval,expected_inf,expected_sup",
        [
            (
                RealInterval(_inf=0.0, _sup=1.0, _interval_type=IntervalType.ClOpen),
                RealInterval(_inf=0.5, _sup=1.5, _interval_type=IntervalType.ClOpen),
                0.5,
                1.0,
            ),
            (
                RealInterval(_inf=0.0, _sup=1.0, _interval_type=IntervalType.ClOpen),
                RealInterval(_inf=1.5, _sup=2.0, _interval_type=IntervalType.ClOpen),
                1.5,
                1.0,
            ),
            (
                RealInterval(_inf=0.0, _sup=1.0, _interval_type=IntervalType.ClOpen),
                RealInterval(_inf=-0.5, _sup=0.5, _interval_type=IntervalType.ClOpen),
                0.0,
                0.5,
            ),
        ],
    )
    def test_intersection(
        self, 
        left_interval, 
        right_interval, 
        expected_inf, 
        expected_sup
    ) -> None:
        inters = intersection(left_interval, right_interval)
        assert inters is not None
        assert inters.inf == expected_inf
        assert inters.sup == expected_sup
        
    def test_interval_commutativity(self, real_intervals) -> None:
        left_interval, right_interval = real_intervals
        
        inters_1 = intersection(left_interval, right_interval)
        inters_2 = intersection(right_interval, left_interval)
        assert inters_1.inf == inters_2.inf
        assert inters_1.sup == inters_2.sup
        assert inters_1.interval_type == inters_2.interval_type
        assert inters_1.length == inters_2.length

    def test_intersection_no_overlap(self, interval_type) -> None:
        left_interval = RealInterval(_inf=0.0, _sup=1.0, _interval_type=interval_type)
        right_interval = RealInterval(_inf=1.5, _sup=2.0, _interval_type=interval_type)

        inters = intersection(left_interval, right_interval)
        assert inters.length == 0.0
        assert inters.inf == pytest.approx(1.5)
        assert inters.sup == pytest.approx(1.0)
        assert inters.interval_type == interval_type
        
    def test_intersection_full_overlap(self, interval_type) -> None:
        left_interval = RealInterval(_inf=0.0, _sup=1.0, _interval_type=interval_type)
        right_interval = RealInterval(_inf=0.0, _sup=1.0, _interval_type=interval_type)

        inters = intersection(left_interval, right_interval)
        assert inters.length == 1.0
        assert inters.inf == pytest.approx(0.0)
        assert inters.sup == pytest.approx(1.0)
        assert inters.interval_type == interval_type
        
    def test_intersection_point_overlap(self, interval_type) -> None:
        left_interval = RealInterval(_inf=0.0, _sup=1.0, _interval_type=interval_type)
        right_interval = RealInterval(_inf=1.0, _sup=2.0, _interval_type=interval_type)

        inters = intersection(left_interval, right_interval)
        assert inters.length == 0.0
        assert inters.inf == pytest.approx(1.0)
        assert inters.sup == pytest.approx(1.0)
        assert inters.interval_type == interval_type
    
    def test_intersection_different_types(self) -> None:
        left_interval = RealInterval(_inf=0.0, _sup=1.0, _interval_type=IntervalType.ClOpen)
        right_interval = RealInterval(_inf=0.5, _sup=1.5, _interval_type=IntervalType.OpenCl)

        with pytest.raises(TypeError):
            intersection(left_interval, right_interval)
    
    def test_intersection_mixed_intervals(self, interval_type) -> None:
        left_partition = Partition(_endpoints=[0.0, 1.0], _interval_type=interval_type)
        right_interval = RealInterval(_inf=0.5, _sup=1.5, _interval_type=interval_type)
        inters = intersection(left_partition, right_interval)
        assert inters.length == 0.5
        assert inters.inf == pytest.approx(0.5)
        assert inters.sup == pytest.approx(1.0)
        assert inters.interval_type == interval_type
            
    def test_intersection_mixed_intervals_2(self) -> None:
        left_interval = RealInterval(_inf=0.0, _sup=1.0, _interval_type=IntervalType.ClOpen)
        right_interval = DyadicInterval(k=3, n=1, _interval_type=IntervalType.ClOpen)

        # TODO: This is a bit of an odd case - we could potentially support this by converting the DyadicInterval to a 
        # RealInterval.
        with pytest.raises(ValueError):
            intersection(left_interval, right_interval)
    
    def test_intersection_invalid_types(self, interval_type) -> None:
        left_interval = RealInterval(_inf=1.0, _sup=0.0, _interval_type=interval_type)

        with pytest.raises(TypeError):
            intersection(left_interval, 0.5)


@pytest.fixture()
def partition(interval_type) -> Partition:
    return Partition(_endpoints=[0.0, 0.5, 1.0], _interval_type=interval_type)


class TestPartition:
    def test_partition_endpoints_and_type(self, partition, interval_type):
        p = partition
        assert p.interval_type is interval_type
        assert p.inf == pytest.approx(0.0)
        assert p.sup == pytest.approx(1.0)
        assert len(p) == 2
        assert p.length == pytest.approx(1.0)

    def test_partition_to_intervals(self, partition, interval_type):
        p = partition
        intervals = p.to_intervals()
        assert len(intervals) == 2
        assert intervals[0].inf == pytest.approx(0.0)
        assert intervals[0].sup == pytest.approx(0.5)
        assert intervals[1].inf == pytest.approx(0.5)
        assert intervals[1].sup == pytest.approx(1.0)
        assert intervals[0].interval_type is interval_type
        assert intervals[1].interval_type is interval_type
        
    def test_partition_short_length(self, interval_type):
        p = Partition(_endpoints=[0.0], _interval_type=interval_type)
        assert len(p) == 0
        assert p.length == pytest.approx(0.0)
        
    def test_partition_to_string(self, partition):
        p = Partition(_endpoints=[0.0, 1.0], _interval_type=IntervalType.ClOpen)
        assert str(p) == "[0.0, 1.0)"
        
        p2 = Partition(_endpoints=[0.0, 1.0], _interval_type=IntervalType.OpenCl)
        assert str(p2) == "(0.0, 1.0]"
        
class TestPartitionMergeAndTruncate:
    def test_partition_merge_aligned(self, partition, interval_type):
        p1 = partition
        p2 = Partition(_endpoints=[0.5, 1.0], _interval_type=interval_type)
        merged = Partition.merge(p1, p2)
        assert isinstance(merged, Partition)
        assert merged.inf == pytest.approx(0.0)
        assert merged.sup == pytest.approx(1.0)
        assert len(merged) == 2
        assert merged.interval_type is interval_type
        
    def test_partition_merge_unaligned(self, partition, interval_type):
        p1 = partition
        p2 = Partition(_endpoints=[0.25, 0.75], _interval_type=interval_type)
        merged = Partition.merge(p1, p2)
        assert isinstance(merged, Partition)
        assert merged.inf == pytest.approx(0.0)
        assert merged.sup == pytest.approx(1.0)
        assert len(merged) == 4
        assert merged.interval_type is interval_type
        
    def test_partition_truncate(self, partition, interval_type):
        p = partition
        ri = RealInterval(_inf=0.25, _sup=0.75, _interval_type=interval_type)
        truncated = Partition.truncate(p, ri)
        assert isinstance(truncated, Partition)
        assert truncated.inf == pytest.approx(0.25)
        assert truncated.sup == pytest.approx(0.75)
        assert len(truncated) == 2
        assert truncated.interval_type is interval_type

class TestPartitionIntersection:
    def test_partition_intersection_interval(self, interval_type):
        p1 = Partition(
            _endpoints=[0.0, 0.5, 1.0], _interval_type=interval_type
        )
        p2 = RealInterval(
            _inf=-0.25, _sup=0.75, _interval_type=interval_type
        )

        inters = intersection(p1, p2)
        assert inters.length > 0.0
        assert inters.inf == pytest.approx(0.0)
        assert inters.sup == pytest.approx(0.75)
        assert len(inters) == 2

    def test_partition_intersection_no_overlap(self, interval_type):
        p1 = Partition(
            _endpoints=[0.0, 0.5, 1.0], _interval_type=interval_type
        )
        p2 = RealInterval(_inf=1.5, _sup=2.0, _interval_type=interval_type)

        inters = intersection(p1, p2)
        # A no-intersection no longer returns None.
        assert inters is not None
        assert inters.length == 0.0

    def test_partition_intersection_subinterval(self, interval_type):
        p1 = Partition(
            _endpoints=[0.0, 0.5, 1.0], _interval_type=interval_type
        )
        p2 = RealInterval(
            _inf=0.25, _sup=0.75, _interval_type=interval_type
        )

        inters = intersection(p1, p2)
        assert inters.length > 0.0
        assert inters.inf == pytest.approx(0.25)
        assert inters.sup == pytest.approx(0.75)
        assert len(inters) == 2
