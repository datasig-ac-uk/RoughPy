
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest

from roughpy import compute as rpc




def test_adjoint_left_ft_mul_identity():

    basis = rpc.TensorBasis(2, 2)

    A_data = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    A = rpc.FreeTensor(A_data, basis)

    rng = np.random.default_rng()
    S = rpc.ShuffleTensor(rng.uniform(-1, 1, size=basis.size()), basis)

    R = rpc.ft_adjoint_left_mul(A, S)

    assert_array_equal(R.data, S.data)


def test_adjoint_left_ft_mul_letter():
    basis = rpc.TensorBasis(2, 2)

    A_data = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    A = rpc.FreeTensor(A_data, basis)

    rng = np.random.default_rng()

    S = rpc.ShuffleTensor(rng.uniform(-1, 1, size=basis.size()), basis)

    R = rpc.ft_adjoint_left_mul(A, S)

    expected_data = np.array([S.data[1], S.data[3], S.data[4], 0.0, 0.0, 0.0, 0.0])

    assert_array_equal(R.data, expected_data)