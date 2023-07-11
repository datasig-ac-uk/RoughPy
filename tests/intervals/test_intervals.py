import pytest

from roughpy import IntervalType, RealInterval


@pytest.fixture(params=[IntervalType.Clopen])
def interval_type(request):
    return request.param


@pytest.mark.parametrize("inf, sup", [(0.0, 1.0), (-5.0, 2.0),
                                      (-float("inf"), float("inf"))])
def test_interval_repr(inf, sup, interval_type):
    ivl = RealInterval(inf, sup, interval_type)
    type_str = "clopen" if interval_type == IntervalType.Clopen else "opencl"

    assert repr(
        ivl) == f"RealInterval({inf=:0.6f}, {sup=:0.6f}, type={type_str})"


def test_interval_equality_unit_intervals(interval_type):
    i1 = RealInterval(0.0, 1.0, interval_type)
    i2 = RealInterval(0.0, 1.0, interval_type)

    assert i1 == i2


@pytest.mark.skip("OpenCL disabled")
def test_interval_equality_fails_different_types():
    i1 = RealInterval(0.0, 1.0, IntervalType.Clopen)
    i2 = RealInterval(0.0, 1.0, IntervalType.Opencl)

    assert i1 != i2


def test_interval_equality_fails_mismatched_endpoints(interval_type):
    i1 = RealInterval(-1.0, 1.0, interval_type)
    i2 = RealInterval(0.0, 2.0, interval_type)

    assert i1 != i2


def test_intersects_with_same_interval(interval_type):
    i1 = RealInterval(0.0, 1.0, interval_type)
    i2 = RealInterval(0.0, 1.0, interval_type)

    assert i1.intersects_with(i2)


def test_intersects_with_overlapping(interval_type):
    i1 = RealInterval(0.0, 1.0, interval_type)
    i2 = RealInterval(-1.0, 0.5, interval_type)

    assert i1.intersects_with(i2)


@pytest.mark.skip("OpenCL disabled")
def test_intersects_with_common_endpoints():
    i1 = RealInterval(0.0, 1.0, IntervalType.Clopen)
    i2 = RealInterval(-1.0, 0.0, IntervalType.Opencl)

    # (-1.0, 0.0] [0.0, 1.0)
    assert i1.intersects_with(i2)


def test_intersects_with_fails_disjoint(interval_type):
    i1 = RealInterval(0.0, 1.0, interval_type)
    i2 = RealInterval(2.0, 3.0, interval_type)

    assert not i1.intersects_with(i2)


def test_intersects_with_fails_common_endpoints_same_type(interval_type):
    i1 = RealInterval(0.0, 1.0, interval_type)
    i2 = RealInterval(-1.0, 0.0, interval_type)

    # [-1.0, 0.0) [0.0, 1.0) or (-1.0, 0.0] (0.0, 1.0]
    assert not i1.intersects_with(i2)


@pytest.mark.skip("OpenCL disabled")
def test_intersects_with_fails_common_endpoint_opposite_type():
    i1 = RealInterval(0.0, 1.0, IntervalType.Opencl)
    i2 = RealInterval(-1.0, 0.0, IntervalType.Clopen)

    # [-1.0, 0.0) (0.0, 1.0]
    assert not i1.intersects_with(i2)
