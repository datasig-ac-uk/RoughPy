

import pytest
import numpy as np

import roughpy.compute as rpc


def test_created_degree_begins():
    width = 5
    depth = 5

    basis = rpc.TensorBasis(width, depth)

    db = getattr(basis, "degree_begin", None)
    assert isinstance(db, np.ndarray)

    assert db.shape == (depth+2, )

    assert db[0] == 0

    size = 1
    for d in range(depth+2)