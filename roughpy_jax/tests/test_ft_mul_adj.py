
import pytest

import jax
import jax.numpy as jnp

from numpy.testing import assert_array_almost_equal, assert_array_equal

import roughpy_jax as rpj



def test_adjoint_left_ft_mul_identity():

    basis = rpj.TensorBasis(2, 2)

    A_data = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

    A = rpj.DenseFreeTensor(A_data, basis)

    rng = jax.random.key(12345)
    S = rpj.DenseShuffleTensor(
        jax.random.uniform(rng, minval=-1.0, maxval=1.0, shape=(basis.size(),)),
        basis
    )

    R = rpj.ft_adjoint_left_mul(A, S)

    assert_array_equal(R.data, S.data)


def test_adjoint_left_ft_mul_letter():
    basis = rpj.TensorBasis(2, 2)

    A_data = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

    A = rpj.DenseFreeTensor(A_data, basis)

    rng = jax.random.key(12345)
    S = rpj.DenseShuffleTensor(
        jax.random.uniform(rng, minval=-1.0, maxval=1.0, shape=(basis.size(),)),
        basis
    )

    R = rpj.ft_adjoint_left_mul(A, S)

    expected_data = jnp.array([S.data[1], S.data[3], S.data[4], 0.0, 0.0, 0.0, 0.0])

    assert_array_equal(R.data, expected_data)