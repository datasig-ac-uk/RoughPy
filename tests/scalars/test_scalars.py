


import pytest
import roughpy as rp


@pytest.fixture
def borrowed_scalar():
    tensor = rp.FreeTensor([1.0, 2.0, 3.0], width=2, depth=1)
    yield tensor[rp.TensorKey(width=2,depth=1)]


def test_add_borrowed_scalar_to_borrowed_scalar(borrowed_scalar):
    result = borrowed_scalar + borrowed_scalar
    assert result == 2.0


def test_add_borrowed_scalar_to_int(borrowed_scalar):
    result = borrowed_scalar + 1
    assert result == 2.0


def test_add_int_to_borrowed_scalar(borrowed_scalar):
    result = 1 + borrowed_scalar
    assert result == 2.0


def test_add_borrowed_scalar_to_float(borrowed_scalar):
    result = borrowed_scalar + 1.
    assert result == 2.0


def test_add_borrowed_to_int_0(borrowed_scalar):
    result = borrowed_scalar + 0
    assert result == borrowed_scalar


def test_add_int_0_to_borrowed_scalar(borrowed_scalar):
    result = 0 + borrowed_scalar
    assert result == borrowed_scalar


def test_add_float_to_borrowed_scalar(borrowed_scalar):
    result = 1. + borrowed_scalar
    assert result == 2.0


def test_multiply_borrowed_scalar_by_float(borrowed_scalar):
    result = borrowed_scalar * 2.
    assert result == 2.0


def test_multiply_borrowed_scalar_by_borrowed_scalar(borrowed_scalar):
    result = borrowed_scalar * borrowed_scalar
    assert result == 1.0


def test_multiply_borrowed_scalar_by_int(borrowed_scalar):
    result = borrowed_scalar * 2
    assert result == 2.0


def test_multiply_borrowed_scalar_by_float(borrowed_scalar):
    result = borrowed_scalar * 2.0
    assert result == 2.0


def test_multiply_int_by_borrowed_scalar(borrowed_scalar):
    result = 2 * borrowed_scalar
    assert result == 2.0


def test_mulitiply_float_by_borrowed_scalar(borrowed_scalar):
    result = 2. * borrowed_scalar
    assert result == 2.0
