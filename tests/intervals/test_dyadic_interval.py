import pickle

import pytest

import roughpy as rp

from roughpy import DyadicInterval, Dyadic


def test_dyadic_interval_pickle_roundtrip():
    d = DyadicInterval(17, 3)

    data = pickle.dumps(d)
    d2 = pickle.loads(data)

    assert d == d2


def test_dincluded_end_Clopen():
    di = DyadicInterval()

    expected = Dyadic(0, 0)

    # print(f"{di=!s}")
    # print(f"{expected=!s}")

    assert rp.Dyadic.dyadic_equals(di.dyadic_included_end(), expected)

@pytest.mark.skip("Fails because first argument is not dyadic. Leave for now. Design difference between cpp and py")
def test_dexcluded_end_Clopen():
    di = DyadicInterval()

    expected = Dyadic(1, 0)

    assert rp.Dyadic.dyadic_equals(di.dyadic_excluded_end(), expected)

@pytest.mark.skip("Opencl not yet available.")
def test_dincluded_end_Opencl():
    di = DyadicInterval(rp.IntervalType.Clopen)

    expected = Dyadic(0, 0)

    assert rp.Dyadic.dyadic_equals(di.dyadic_included_end(), expected)

@pytest.mark.skip("Opencl not yet available.")
def test_dexcluded_end_Opencl():
    di = DyadicInterval(rp.IntervalType.Clopen)
    expected = Dyadic(-1, 0)

    assert rp.Dyadic.dyadic_equals(di.dyadic_excluded_end(), expected)

def test_included_end_Clopen():
    di = DyadicInterval()

    assert di.included_end() == 0.0


def test_excluded_end_Clopen():
    di = DyadicInterval()

    assert di.excluded_end() == 1.0

@pytest.mark.skip("Opencl not yet available.")
def test_included_end_Opencl():
    di = DyadicInterval(rp.IntervalType.Clopen)

    assert di.included_end() == 0.0

@pytest.mark.skip("Opencl not yet available.")
def test_excluded_end_Opencl():
    di = DyadicInterval(rp.IntervalType.Clopen)

    assert di.excluded_end() == -1.0


def test_dsup_Clopen():
    di = DyadicInterval()
    expected = Dyadic(1, 0)

    assert rp.Dyadic.dyadic_equals(di.dyadic_sup(), expected)


def test_dinf_Clopen():
    di = DyadicInterval()
    expected = Dyadic(0, 0)

    assert rp.Dyadic.dyadic_equals(di.dyadic_inf(), expected)

@pytest.mark.skip("Opencl not yet available.")
def test_dsup_Opencl():
    di = DyadicInterval(rp.IntervalType.Clopen)
    expected = Dyadic(0, 0)

    assert rp.Dyadic.dyadic_equals(di.dyadic_sup(), expected)

@pytest.mark.skip("Opencl not yet defined.")
def test_dinf_Opencl():
    di = DyadicInterval(rp.IntervalType.Clopen)
    expected = Dyadic(-1, 0)

    assert rp.Dyadic.dyadic_equals(di.inf(), expected)


def test_sup_Clopen():
    di = DyadicInterval()

    assert di.sup() == 1.0


def test_inf_Clopen():
    di = DyadicInterval()

    assert di.inf() == 0.0

@pytest.mark.skip("Opencl not yet available.")
def test_sup_Opencl():
    di = DyadicInterval(rp.IntervalType.Clopen)

    assert di.sup() == 0.0

@pytest.mark.skip("Opencl not yet available.")
def test_inf_Opencl():
    di = DyadicInterval(rp.IntervalType.Clopen)

    assert di.inf() == -1.0

@pytest.mark.skip("flip_interval not yet available")
def test_flip_interval_aligned_Clopen():

    di = DyadicInterval(Dyadic(0,1))
    expected = Dyadic(1,1)

    assert di.flip_interval() == expected

@pytest.mark.skip("flip_interval not yet available.")
def test_flip_interval_non_aligned_Clopen():

    di = DyadicInterval(Dyadic(1,1))
    expected = Dyadic(0,1)

    assert di.flip_interval() == expected

@pytest.mark.skip("Opencl and flip_interval not yet available.")
def test_flip_interval_aligned_Opencl():

    di = DyadicInterval(Dyadic(0,1), rp.IntervalType.Clopen)
    expected = Dyadic(-1,-1, rp.IntervalType.Clopen)

    assert di.flip_interval() == expected

@pytest.mark.skip("Opencl and flip_interval not yet available.")
def test_flip_interval_non_aligned_Opencl():

    di = DyadicInterval(Dyadic(1,1), rp.IntervalType.Clopen)
    expected = Dyadic(2,1, rp.IntervalType.Clopen)

    assert di.flip_interval() == expected

@pytest.mark.skip("aligned not yet available.")
def test_aligned_aligned():

    di = DyadicInterval(0, 0)

    assert di.aligned()

@pytest.mark.skip("aligned not yet available.")
def test_aligned_non_aligned():

    di = DyadicInterval(1, 0)

    assert not di.aligned()


def test_contains_unit_and_half():

    parent, child = DyadicInterval(Dyadic(0, 0)), DyadicInterval(Dyadic(0, 1))

    assert parent.contains(child)

def test_contains_unit_and_half_compliment():

    parent, child = DyadicInterval(Dyadic(0, 0)), DyadicInterval(Dyadic(1, 1))

    assert parent.contains(child)

def test_contains_unit_and_unit():

    parent, child =DyadicInterval( Dyadic(0, 0)), DyadicInterval(Dyadic(0, 0))

    assert parent.contains(child)


def test_contains_unit_and_longer():

    parent, child = DyadicInterval(Dyadic(0, 0)), DyadicInterval(Dyadic(0, -1))

    assert not parent.contains(child)

def test_contains_unit_disjoint():

    parent, child = DyadicInterval(Dyadic(0, 0)), DyadicInterval(Dyadic(1, 0))

    assert not parent.contains(child)

def test_contains_unit_disjoint_and_shorter_right():

    parent, child = DyadicInterval(Dyadic(0, 0)), DyadicInterval(Dyadic(2, 1))

    assert not parent.contains(child)

def test_contains_unit_disjoint_and_shorter_left():

    parent, child = DyadicInterval(Dyadic(0, 0)), DyadicInterval(Dyadic(-1, 1))

    assert not parent.contains(child)

def test_to_dyadic_intervals_unit_interval_tol_1():

    intervals = rp.DyadicInterval.to_dyadic_intervals(rp.RealInterval(0.0,1.0), 1)
    expected = DyadicInterval(Dyadic(0,0))

    assert len(intervals) == 1 and intervals[0] == expected

def test_to_dyadic_intervals_unit_interval_tol_5():
    intervals = rp.DyadicInterval.to_dyadic_intervals(rp.RealInterval(0,1.0), 5)
    expected = DyadicInterval(Dyadic(0,0))

    assert len(intervals) == 1 and intervals[0] == expected

def test_to_dyadic_intervals_mone_one_interval_tol_1():

    intervals = rp.DyadicInterval.to_dyadic_intervals(rp.RealInterval(-1.0,1.0), 1)
    expected0, expected1 = rp.DyadicInterval(rp.Dyadic(-1,0)) , rp.DyadicInterval(Dyadic(0,0))

    assert len(intervals) == 2 and intervals[0] == expected0 and intervals[1] == expected1


def test_to_dyadic_intervals_mone_onehalf_interval_tol_1():
    intervals = rp.DyadicInterval.to_dyadic_intervals(rp.RealInterval(-1.0,1.5), 1)
    expected0, expected1, expected2 = rp.DyadicInterval(rp.Dyadic(-1,0)) , rp.DyadicInterval(Dyadic(0,0)), rp.DyadicInterval(Dyadic(2,1))

    assert len(intervals) == 3 and intervals[0] == expected0 and intervals[1] == expected1 and intervals[2] == expected2


def test_to_dyadic_intervals_mone_onequarter_interval_tol_1():
    intervals = rp.DyadicInterval.to_dyadic_intervals(rp.RealInterval(-1.0,1.25), 1)
    expected0, expected1 = rp.DyadicInterval(rp.Dyadic(-1,0)) , rp.DyadicInterval(Dyadic(0,0))


    assert len(intervals) == 2 and intervals[0] == expected0 and intervals[1] == expected1 , f"{intervals}!={expected0}"

def test_to_dyadic_intervals_mone_onequarter_interval_tol_2():
    intervals = rp.DyadicInterval.to_dyadic_intervals(rp.RealInterval(-1.0,1.25), 2)

    expected0, expected1, expected2 = rp.DyadicInterval(rp.Dyadic(-1,0)), rp.DyadicInterval(Dyadic(0,0)), rp.DyadicInterval(Dyadic(4,2))

    assert len(intervals) == 3 and intervals[0] == expected0 and intervals[1] == expected1 and intervals[2] == expected2

def test_to_dyadic_intervals_0_upper_interval_tol_1():
    intervals = rp.DyadicInterval.to_dyadic_intervals(rp.RealInterval(0.0, 1.63451), 1)

    expected0, expected1 = rp.DyadicInterval(rp.Dyadic(0,0)) , rp.DyadicInterval(Dyadic(2,1))

    assert len(intervals) == 2 and intervals[0] == expected0 and intervals[1] == expected1

def test_to_dyadic_intervals_0_upper_interval_tol_2():

    intervals = rp.DyadicInterval.to_dyadic_intervals(rp.RealInterval(0.0, 1.63451), 2)

    expected0, expected1 = rp.DyadicInterval(rp.Dyadic(0,0)) , rp.DyadicInterval(Dyadic(2,1))

    assert len(intervals) == 2 and intervals[0] == expected0 and intervals[1] == expected1

def test_to_dyadic_intervals_0_upper_interval_tol_3():

    intervals = rp.DyadicInterval.to_dyadic_intervals(rp.RealInterval(0.0, 1.63451), 3)

    expected0, expected1, expected2 = rp.DyadicInterval(rp.Dyadic(0,0)) , rp.DyadicInterval(Dyadic(2,1)), rp.DyadicInterval(Dyadic(12,3))

    assert len(intervals) == 3 and intervals[0] == expected0 and intervals[1] == expected1 and intervals[2] == expected2

def test_to_dyadic_intervals_0_upper_interval_tol_7():

    intervals = rp.DyadicInterval.to_dyadic_intervals(rp.RealInterval(0.0, 1.63451), 7)

    expected0, expected1, expected2, expected3 = rp.DyadicInterval(rp.Dyadic(0,0)) , rp.DyadicInterval(Dyadic(2,1)), rp.DyadicInterval(Dyadic(12,3)), rp.DyadicInterval(Dyadic(208,7))

    assert len(intervals) == 4 and intervals[0] == expected0 and intervals[1] == expected1 and intervals[2] == expected2 and intervals[3] == expected3

def test_shrink_interval_left_Clopen():
    start = DyadicInterval(Dyadic(0, 0))
    expected = DyadicInterval(Dyadic(0, 1))

    assert start.shrink_left() == expected

def test_shrink_interval_right_Clopen():
    start = DyadicInterval(Dyadic(0, 0))
    expected = DyadicInterval(Dyadic(1, 1))

    assert start.shrink_right() == expected

@pytest.mark.skip("Opencl not implemented")
def test_shrink_interval_left_Opencl():

    start = DyadicInterval(Dyadic(0, 0), rp.IntervalType.Clopen)
    expected = DyadicInterval(Dyadic(-1, 1), rp.IntervalType.Clopen)

    assert start.shrink_left() == expected

@pytest.mark.skip("Opencl not implemented")
def test_shrink_interval_right_Opencl():

    start = DyadicInterval(Dyadic(0, 0), rp.IntervalType.Clopen)
    expected = DyadicInterval(Dyadic(0, 1), rp.IntervalType.Clopen)


    assert start.shrink_right() == expected

def test_shrink_to_contained_end_Clopen():

    start = DyadicInterval(Dyadic(0, 0))
    expected = DyadicInterval(Dyadic(0, 1))

    assert start.shrink_to_contained_end() == expected

def test_shrink_to_omitted_end_Clopen():

    start = DyadicInterval(Dyadic(0, 0))
    expected = DyadicInterval(Dyadic(1, 1))

    assert start.shrink_to_omitted_end() == expected

@pytest.mark.skip("Opencl not yet available.")
def test_shrink_to_contained_end_Opencl():

    start = DyadicInterval(Dyadic(0, 0), rp.IntervalType.Clopen)
    expected = DyadicInterval(Dyadic(0, 1), rp.IntervalType.Clopen)

    assert start.shrink_to_contained_end() == expected

@pytest.mark.skip("Opencl not yet available.")
def test_shrink_to_omitted_end_Opencl():

    start = DyadicInterval(Dyadic(0, 0), rp.IntervalType.Clopen)
    expected = DyadicInterval(Dyadic(-1, 1), rp.IntervalType.Clopen)

    assert start.shrink_to_omitted_end() == expected



