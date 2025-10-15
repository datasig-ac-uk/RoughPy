import pytest
import numpy as np

from typing import Callable

from numpy.testing import assert_array_almost_equal, assert_array_equal


import roughpy.compute as rpc




@pytest.fixture
def random_shuffle(rng) -> Callable[[int, int], rpc.ShuffleTensor]:
    basis_cache = {}

    def make_shuffle(width: int, depth: int) -> rpc.ShuffleTensor:
        nonlocal basis_cache
        basis = None
        if (width, depth) not in basis_cache:
            basis = basis_cache[(width, depth)] = rpc.TensorBasis(width, depth)
        else:
            basis = basis_cache[(width, depth)]

        data = rng.uniform(-1.0, 1.0, size=(basis.size(),))

        return rpc.ShuffleTensor(data, basis)

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

    result1 = rpc.st_mul(lhs, rhs)
    result2 = rpc.st_mul(rhs, lhs)

    assert_array_almost_equal(result1.data, result2.data)



def test_product_unit(random_shuffle):
    lhs = random_shuffle(4, 3)

    unit_data = np.zeros_like(lhs.data)
    unit_data[0] = 1.
    unit = rpc.ShuffleTensor(unit_data, lhs.basis)

    result1 = rpc.st_mul(lhs, unit)
    result2 = rpc.st_mul(unit, lhs)

    assert_array_almost_equal(result1.data, lhs.data)
    assert_array_almost_equal(result2.data, lhs.data)



def test_product_two_letters():
    basis = rpc.TensorBasis(4, 3)
    to_idx = _word_to_idx_fn(4)

    lhs_data = np.zeros((basis.size(),), dtype=np.float64)
    rhs_data = np.zeros((basis.size(),), dtype=np.float64)

    lhs_data[1] = 1.0
    rhs_data[2] = 1.0

    lhs = rpc.ShuffleTensor(lhs_data, basis)
    rhs = rpc.ShuffleTensor(rhs_data, basis)

    result = rpc.st_mul(lhs, rhs)

    expected = np.zeros_like(lhs.data)
    expected[to_idx(1, 2)] = 1.0
    expected[to_idx(2, 1)] = 1.0

    assert_array_equal(result.data, expected)


def test_product_letter_and_deg2():
    basis = rpc.TensorBasis(4, 3)
    to_idx = _word_to_idx_fn(4)

    lhs_data = np.zeros((basis.size(),), dtype=np.float64)
    rhs_data = np.zeros((basis.size(),), dtype=np.float64)

    lhs_data[to_idx(1, 2)] = 1.0
    rhs_data[3] = 1.0

    lhs = rpc.ShuffleTensor(lhs_data, basis)
    rhs = rpc.ShuffleTensor(rhs_data, basis)

    result = rpc.st_mul(lhs, rhs)

    expected = np.zeros_like(lhs.data)
    expected[to_idx(1, 2, 3)] = 1.0
    expected[to_idx(1, 3, 2)] = 1.0
    expected[to_idx(3, 1, 2)] = 1.0

    assert_array_equal(result.data, expected)

