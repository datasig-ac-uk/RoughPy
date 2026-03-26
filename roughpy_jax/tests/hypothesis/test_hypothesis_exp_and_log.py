"Property-based tests for ft_exp and ft_log"

from numpy.testing import assert_array_almost_equal
from hypothesis import given

import roughpy_jax as rpj
from roughpy_jax.strategies import lie_increment

@given(L=lie_increment())
def test_first_index_is_zero(L):
    assert L.data[0] == 0

@given(L=lie_increment())
def test_exp_and_log_roundtrip(L):
    a = rpj.ft_exp(L)
    expected = rpj.ft_log(a)
    assert_array_almost_equal(expected.data, L.data)
