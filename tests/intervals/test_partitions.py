
import pickle

import numpy as np

import roughpy as rp
from roughpy import Partition, RealInterval


def test_partition_trim_intermediates():
    interval = rp.RealInterval(0.0, 1.0)
    partition = rp.Partition(interval, intermediates=np.arange(10))

    assert partition.intermediates() == []

def test_partition_pickle_roundtrip():

    p = Partition(RealInterval(0.5, 1.6), [0.9, 1.3])

    data = pickle.dumps(p)
    p2 = pickle.loads(data)

    assert p == p2
