
import pickle

import pytest

from roughpy import RealInterval




def test_pickle_real_interval_roundtrip():
    r = RealInterval(0.0, 1.0)

    data = pickle.dumps(r)
    r2 = pickle.loads(data)

    assert r == r2


