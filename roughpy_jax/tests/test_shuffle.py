from typing import Callable

import pytest
import numpy as np
import jax.numpy as jnp
from numpy.testing import assert_array_almost_equal, assert_array_equal

import roughpy_jax as rpj


@pytest.fixture
def random_shuffle() -> Callable[[int, int], rpj.ShuffleTensor]:
    rng = np.random.default_rng(12345)
    basis_cache = {}

    def make_shuffle(width: int, depth: int) -> rpj.ShuffleTensor:
        nonlocal basis_cache
        basis = None
        if (width, depth) not in basis_cache:
            basis = basis_cache[(width, depth)] = rpj.TensorBasis(width, depth)
        else:
            basis = basis_cache[(width, depth)]

        data = rng.uniform(-1.0, 1.0, size=(basis.size(),))

        return rpj.ShuffleTensor(data, basis)

    return make_shuffle


def _word_to_idx_fn(width: int) -> Callable[[...], int]:
    def inner(*letters) -> int:
        idx = 0
        for letter in letters:
            idx = idx * width + letter
        return idx

    return inner


def test_product_commutative(random_shuffle):
    lhs = random_shuffle(4, 3)
    rhs = random_shuffle(4, 3)

    result1 = rpj.st_mul(lhs, rhs)
    result2 = rpj.st_mul(rhs, lhs)

    assert jnp.allclose(result1.data, result2.data)


def test_product_unit(random_shuffle):
    lhs = random_shuffle(4, 3)

    unit_data = np.zeros_like(lhs.data)
    unit_data[0] = 1.
    unit = rpj.ShuffleTensor(unit_data, lhs.basis)

    result1 = rpj.st_mul(lhs, unit)
    result2 = rpj.st_mul(unit, lhs)

    assert jnp.allclose(result1.data, lhs.data)
    assert jnp.allclose(result2.data, lhs.data)


def test_product_two_letters():
    basis = rpj.TensorBasis(4, 3)
    to_idx = _word_to_idx_fn(4)

    lhs_data = np.zeros((basis.size(),), dtype=np.float64)
    rhs_data = np.zeros((basis.size(),), dtype=np.float64)

    lhs_data[1] = 1.0
    rhs_data[2] = 1.0

    lhs = rpj.ShuffleTensor(lhs_data, basis)
    rhs = rpj.ShuffleTensor(rhs_data, basis)

    result = rpj.st_mul(lhs, rhs)

    expected = np.zeros_like(lhs.data)
    expected[to_idx(1, 2)] = 1.0
    expected[to_idx(2, 1)] = 1.0

    assert jnp.allclose(result.data, expected)


def test_product_letter_and_deg2():
    basis = rpj.TensorBasis(4, 3)
    to_idx = _word_to_idx_fn(4)

    lhs_data = np.zeros((basis.size(),), dtype=np.float64)
    rhs_data = np.zeros((basis.size(),), dtype=np.float64)

    lhs_data[to_idx(1, 2)] = 1.0
    rhs_data[3] = 1.0

    lhs = rpj.ShuffleTensor(lhs_data, basis)
    rhs = rpj.ShuffleTensor(rhs_data, basis)

    result = rpj.st_mul(lhs, rhs)

    expected = np.zeros_like(lhs.data)
    expected[to_idx(1, 2, 3)] = 1.0
    expected[to_idx(1, 3, 2)] = 1.0
    expected[to_idx(3, 1, 2)] = 1.0

    assert jnp.allclose(result.data, expected)
