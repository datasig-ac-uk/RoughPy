
import pickle

import pytest

from roughpy import DyadicInterval, Dyadic



def test_dyadic_interval_pickle_roundtrip():

    d = DyadicInterval(17, 3)

    data = pickle.dumps(d)
    d2 = pickle.loads(data)

    assert d == d2
