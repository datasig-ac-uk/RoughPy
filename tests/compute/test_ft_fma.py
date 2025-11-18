import numpy as np
from roughpy import compute
from numpy.random import default_rng
from numpy.testing import assert_array_almost_equal
from fractions import Fraction


def test_dense_ftfma_explicit():
    basis = compute.TensorBasis(2, 2)
    b_data = np.array([2, 1, 3, 0.5, -1, 2, 0], dtype=np.float32)
    c_data = np.array([-1, 4, 0, 1, 1, 0, 2], dtype=np.float32)
    a = compute.FreeTensor(np.zeros(basis.size(), dtype=np.float32), basis)
    b = compute.FreeTensor(b_data, basis)
    c = compute.FreeTensor(c_data, basis)
    compute.ft_fma(a, b, c)

    # zeroth order term: 2 . - 1
    # first order term: 2. (4 0) + -1 . (1 3)
    # and so on
    expected = np.array([-2, 7, -3, 5.5, 3, 10, 4], dtype=np.float32)
    assert_array_almost_equal(a.data, expected)


def test_dense_ftfma_construction():
    rng = default_rng(12345)

    basis = compute.TensorBasis(2, 2)
    a = compute.FreeTensor(np.zeros(basis.size(), dtype=np.float32), basis)
    b = compute.FreeTensor(np.zeros(basis.size(), dtype=np.float32), basis)
    c = compute.FreeTensor(np.zeros(basis.size(), dtype=np.float32), basis)
    a.data = rng.uniform(-1, 1, size=basis.size())
    b.data = rng.uniform(-1, 1, size=basis.size())
    c.data = rng.uniform(-1, 1, size=basis.size())
    expected = a.data.copy()

    # scalar term
    expected[0] += b.data[0] * c.data[0]

    # first order term
    expected[1:3] += b.data[0] * c.data[1:3] + c.data[0] * b.data[1:3]

    # second order term
    expected[3:] += (
        b.data[0] * c.data[3:]
        + c.data[0] * b.data[3:]
        + np.outer(b.data[1:3], c.data[1:3]).flatten()
    )

    compute.ft_fma(a, b, c)
    assert_array_almost_equal(expected, a.data)


def test_dense_ftfma_fractions():
    basis = compute.TensorBasis(2, 2)
    b_data = np.array([Fraction(2, 3), Fraction(1, 2), Fraction(3, 4), Fraction(1, 2),
                       Fraction(-1, 3), Fraction(2, 5), Fraction(0, 1)], dtype=object)
    c_data = np.array([Fraction(-1, 2), Fraction(4, 3), Fraction(0, 1), Fraction(1, 4),
                       Fraction(1, 5), Fraction(0, 1), Fraction(2, 7)], dtype=object)
    a = compute.FreeTensor(np.zeros(basis.size(), dtype=object), basis)
    b = compute.FreeTensor(b_data, basis)
    c = compute.FreeTensor(c_data, basis)

    compute.ft_fma(a, b, c)

    expected = np.array([
        b_data[0] * c_data[0],
        b_data[1] * c_data[0] + b_data[0] * c_data[1],
        b_data[2] * c_data[0] + b_data[0] * c_data[2],
        b_data[3] * c_data[0] + b_data[1] * c_data[1] + b_data[0] * c_data[3],
        b_data[4] * c_data[0] + b_data[1] * c_data[2] + b_data[0] * c_data[4],
        b_data[5] * c_data[0] + b_data[2] * c_data[1] + b_data[0] * c_data[5],
        b_data[6] * c_data[0] + b_data[2] * c_data[2] + b_data[0] * c_data[6],
    ], dtype=object)
    np.testing.assert_array_equal(a.data, expected)
