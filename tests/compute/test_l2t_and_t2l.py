
from roughpy import compute as rpc

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal




def test_l2t_t2l_roundtrip():
    lie_basis = rpc.LieBasis(4, 4)
    tensor_basis = rpc.TensorBasis(lie_basis.width, lie_basis.depth)

    rng = np.random.default_rng()

    x_data = rng.uniform(-1.0, 1.0, (lie_basis.size(),))

    x = rpc.Lie(x_data, lie_basis)

    tensor_x = rpc.lie_to_tensor(x, tensor_basis)

    y = rpc.tensor_to_lie(tensor_x, lie_basis)

    assert_array_almost_equal(x.data, y.data, decimal=15)

