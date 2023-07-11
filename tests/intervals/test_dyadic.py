import itertools
import math

import pytest

from roughpy import Dyadic


@pytest.mark.parametrize("k, n",
                         itertools.product(range(-10, 10), range(0, 10)))
def test_dyadic_to_float(k, n):
    d = Dyadic(k, n)
    assert float(d) == math.ldexp(k, -n)


@pytest.mark.parametrize("n", range(1, 15))
def test_rebase_dyadic(n):
    d = Dyadic(1, 0)

    d.rebase(n)
    assert float(d) == 1.0
    assert d.n == n
    assert d.k == 1 << n
