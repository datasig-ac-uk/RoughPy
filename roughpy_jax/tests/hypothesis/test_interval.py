from hypothesis import given

from roughpy_jax.intervals import IntervalType
from roughpy_jax.strategies import intervals

@given(interval=intervals(IntervalType.ClOpen))
def test_interval(interval):
    assert interval.interval_type == IntervalType.ClOpen
    assert interval.sup > interval.inf
