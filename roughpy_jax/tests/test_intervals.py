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
    assert float(d1) == pytest.approx(1.5)

    d2 = Dyadic(k=-1, n=2)
    assert float(d2) == pytest.approx(-0.25)

    d3 = Dyadic(k=5, n=0)
    assert float(d3) == pytest.approx(5.0)


def test_real_interval_basic_and_frozen():
    ri = RealInterval[float](_inf=0.1, _sup=0.2, _interval_type=IntervalType.ClOpen)
    assert ri.interval_type() is IntervalType.ClOpen
    assert ri.inf() == pytest.approx(0.1)
    assert ri.sup() == pytest.approx(0.2)

    # Frozen dataclass should not allow mutation
    with pytest.raises(Exception):
        setattr(ri, "_inf", 0.0)


def test_partition_endpoints_and_type():
    p = Partition[float](_endpoints=[0.0, 0.5, 1.0], _interval_type=IntervalType.OpenCl)
    assert p.interval_type() is IntervalType.OpenCl
    assert p.inf() == pytest.approx(0.0)
    assert p.sup() == pytest.approx(1.0)


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
    assert p.inf() == pytest.approx(0.0)
    assert p.sup() == pytest.approx(1.0)
