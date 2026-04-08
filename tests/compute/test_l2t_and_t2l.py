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


def test_l2t_scaled_t2l_roundtrip():
    lie_basis = rpc.LieBasis(4, 4)
    tensor_basis = rpc.TensorBasis(lie_basis.width, lie_basis.depth)

    rng = np.random.default_rng()

    x_data = rng.uniform(-1.0, 1.0, (lie_basis.size(),))

    x = rpc.Lie(x_data, lie_basis)

    tensor_x = rpc.lie_to_tensor(x, tensor_basis, scale_factor=0.5)

    y = rpc.tensor_to_lie(tensor_x, lie_basis)

    assert_array_almost_equal(0.5 * x.data, y.data, decimal=15)


def test_l2t_t2l_scaled_roundtrip():
    lie_basis = rpc.LieBasis(4, 4)
    tensor_basis = rpc.TensorBasis(lie_basis.width, lie_basis.depth)

    rng = np.random.default_rng()

    x_data = rng.uniform(-1.0, 1.0, (lie_basis.size(),))

    x = rpc.Lie(x_data, lie_basis)

    tensor_x = rpc.lie_to_tensor(x, tensor_basis)

    y = rpc.tensor_to_lie(tensor_x, lie_basis, scale_factor=0.5)

    assert_array_almost_equal(0.5 * x.data, y.data, decimal=15)


def test_l2t_t2l_caching_consistency():
    """
    Bug reported in #314

    t2l was cached in l2t cache dict
    """

    lie_basis = rpc.LieBasis(4, 4)

    l2t_1 = lie_basis.get_l2t_matrix(np.dtype("float64"))
    t2l_1 = lie_basis.get_t2l_matrix(np.dtype("float64"))

    l2t_2 = lie_basis.get_l2t_matrix(np.dtype("float64"))
    t2l_2 = lie_basis.get_t2l_matrix(np.dtype("float64"))

    assert l2t_1 is l2t_2
    assert t2l_1 is t2l_2
