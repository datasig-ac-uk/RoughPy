


from roughpy import compute as rpc


import numpy as np
import pytest

from numpy.testing import assert_array_equal, assert_array_almost_equal


def test_dense_ft_exp_zero():

    basis = rpc.TensorBasis(2, 2)

    a = rpc.FreeTensor(np.zeros(basis.size(), dtype=np.float64), basis)

    exp_a = rpc.ft_exp(a)

    expected_data = np.zeros_like(a.data)
    expected_data[0] = 1.0

    assert_array_equal(exp_a.data, expected_data)


def test_dense_ft_exp_letter():

    basis = rpc.TensorBasis(2, 2)
    a = rpc.FreeTensor(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), basis)

    exp_a = rpc.ft_exp(a)

    expected_data = np.array([1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0])

    assert_array_equal(exp_a.data, expected_data)


def test_dense_ft_log_identity():
    basis = rpc.TensorBasis(2, 2)
    a = rpc.FreeTensor(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), basis)

    log_a = rpc.ft_log(a)

    expected = np.zeros_like(a.data)
    assert_array_equal(log_a.data, expected)


def test_dense_ft_exp_log_roundtrip():

    basis = rpc.TensorBasis(2, 2)
    a = rpc.FreeTensor(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), basis)

    rng = np.random.default_rng()
    a.data[1:basis.width+1] = rng.normal(size=(basis.width,))

    exp_a = rpc.ft_exp(a)
    log_exp_a = rpc.ft_log(exp_a)

    assert_array_almost_equal(log_exp_a.data, a.data)


def test_dense_ft_exp_and_fmexp():

    basis = rpc.TensorBasis(2, 2)
    e = rpc.FreeTensor(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), basis)


    rng = np.random.default_rng()

    x = rpc.FreeTensor(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), basis)
    x.data[1:basis.width+1] = rng.normal(size=(basis.width,))

    e_exp_b = rpc.ft_fmexp(e, x)
    exp_b = rpc.ft_exp(x)

    assert_array_almost_equal(e_exp_b.data, exp_b.data)


def test_dense_ft_fmexp():
    basis = rpc.TensorBasis(2, 2)
    a = rpc.FreeTensor(np.array([1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0]), basis)

    rng = np.random.default_rng()
    x = rpc.FreeTensor(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), basis)
    x.data[1:basis.width+1] = rng.normal(size=(basis.width,))

    b = rpc.ft_fmexp(a, x)

    expected = rpc.ft_mul(a, rpc.ft_exp(x))
    assert_array_almost_equal(b.data, expected.data)
