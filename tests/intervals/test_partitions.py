
import pickle

import pytest

from roughpy import Partition, RealInterval

def test_partition_pickle_roundtrip():

    p = Partition(RealInterval(0.5, 1.6), [0.9, 1.3])

    data = pickle.dumps(p)
    p2 = pickle.loads(data)

    assert p == p2