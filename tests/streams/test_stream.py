import pickle


import pytest
import numpy as np

from roughpy import LieIncrementStream, DPReal


# @pytest.mark.xfail("currently not implemented correctly")
def test_stream_pickle_roundtrip():

    data = np.array([
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ])

    s = LieIncrementStream.from_increments(data, width=3, depth=4, dtype=DPReal)

    s2 = pickle.loads(pickle.dumps(s))

    assert s2.signature() == s.signature()
