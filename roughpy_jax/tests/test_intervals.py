import math
import pytest

from roughpy_jax.intervals import (
    Interval, 
    IntervalType, 
    Dyadic,
    DyadicInterval, 
    RealInterval, 
    Partition
) 


def test_dyadic_float_conversion_basic():
    d1 = Dyadic(k=3, n=1)
    assert float(d1) == 1.5

    d2 = Dyadic(k=-1, n=2)
    assert float(d2) == -0.25

    d3 = Dyadic(k=5, n=0)
    assert float(d3) == 5.0


def test_real_interval_basic_and_frozen():
    ri = RealInterval[float](_inf=0.1, _sup=0.2, _interval_type=IntervalType.ClOpen)
    assert ri.interval_type is IntervalType.ClOpen
    assert ri.inf == pytest.approx(0.1)
    assert ri.sup == pytest.approx(0.2)

    # Frozen dataclass should not allow mutation
    with pytest.raises(Exception):
        setattr(ri, "_inf", 0.0)
        

def test_protocol_conformance_runtime_checkable():
    # All three concrete types should be instances of the Interval protocol
    di = DyadicInterval(k=2, n=1, _interval_type=IntervalType.ClOpen)
    ri = RealInterval[float](_inf=1.0, _sup=2.0, _interval_type=IntervalType.OpenCl)
    p = Partition[float](_endpoints=[0.0, 1.0], _interval_type=IntervalType.ClOpen)

    assert isinstance(di, Interval)
    assert isinstance(ri, Interval)
    assert isinstance(p, Interval)


def test_partition_with_dyadic_endpoints_uses_float_conversion():
    endpoints = [Dyadic(k=0, n=0), Dyadic(k=1, n=0)]
    p = Partition[Dyadic](_endpoints=endpoints, _interval_type=IntervalType.ClOpen)
    assert p.inf == pytest.approx(0.0)
    assert p.sup == pytest.approx(1.0)
    

@pytest.mark.parametrize("left_interval,right_interval,expected_inf,expected_sup", [
    (
        RealInterval[float](_inf=0.0, _sup=1.0, _interval_type=IntervalType.ClOpen),
        RealInterval[float](_inf=0.5, _sup=1.5, _interval_type=IntervalType.ClOpen),
        0.5, 1.0
    ),
    (
        DyadicInterval(k=0, n=0, _interval_type=IntervalType.ClOpen),
        DyadicInterval(k=1, n=0, _interval_type=IntervalType.ClOpen),
        0.5, 1.0
    ),
    (
        Partition[float](_endpoints=[0.0, 1.0], _interval_type=IntervalType.ClOpen),
        Partition[float](_endpoints=[0.5, 1.5], _interval_type=IntervalType.ClOpen),
        0.5, 1.0
    )
])
def test_intersection(left_interval, right_interval, expected_inf, expected_sup):
    intersection = left_interval.intersection(right_interval)
    assert intersection is not None
    assert intersection.inf == expected_inf
    assert intersection.sup == expected_sup
    
    
class TestPartition:
    def test_partition_endpoints_and_type(self):
        p = Partition[float](_endpoints=[0.0, 0.5, 1.0], _interval_type=IntervalType.OpenCl)
        assert p.interval_type is IntervalType.OpenCl
        assert p.inf == pytest.approx(0.0)
        assert p.sup == pytest.approx(1.0)
        assert len(p) == 2
        
    def test_partition_intersection_interval(self):
        p1 = Partition[float](_endpoints=[0.0, 0.5, 1.0], _interval_type=IntervalType.OpenCl)
        p2 = RealInterval[float](_inf=-0.1, _sup=0.75, _interval_type=IntervalType.OpenCl)
        
        intersection = p1.intersection(p2)
        assert intersection is not None
        assert intersection.inf == pytest.approx(0.0)
        assert intersection.sup == pytest.approx(0.75)
        assert len(intersection) == 2
        
    def test_partition_intersection_no_overlap(self):
        p1 = Partition[float](_endpoints=[0.0, 0.5, 1.0], _interval_type=IntervalType.OpenCl)
        p2 = RealInterval[float](_inf=1.5, _sup=2.0, _interval_type=IntervalType.OpenCl)
        
        intersection = p1.intersection(p2)
        assert intersection is None
    
    def test_partition_intersection_subinterval(self):
        p1 = Partition[float](_endpoints=[0.0, 0.5, 1.0], _interval_type=IntervalType.OpenCl)
        p2 = RealInterval[float](_inf=0.25, _sup=0.75, _interval_type=IntervalType.OpenCl)
        
        intersection = p1.intersection(p2)
        assert intersection is not None
        assert intersection.inf == pytest.approx(0.25)
        assert intersection.sup == pytest.approx(0.75)
        assert len(intersection) == 2
