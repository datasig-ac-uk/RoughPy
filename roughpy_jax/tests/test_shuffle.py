from typing import Callable
import pytest
import numpy as np
import jax.numpy as jnp
import roughpy_jax as rpj


def _word_to_idx_fn(width: int) -> Callable[[...], int]:
    def inner(*letters) -> int:
        idx = 0
        for letter in letters:
            idx = idx * width + letter
        return idx

    return inner


def test_shuffle_dense_st_fma_array_mismatch(rpj_test_fixture_type_mismatch):
    f = rpj_test_fixture_type_mismatch

    # Mismatch first and second widths
    with pytest.raises(ValueError):
        rpj.st_fma(f.st_f32(2, 2), f.st_f32(3, 2), f.st_f32(2, 2))

    # Mismatch first and third widths
    with pytest.raises(ValueError):
        rpj.st_fma(f.st_f32(2, 2), f.st_f32(2, 2), f.st_f32(3, 2))

    # FIXME for review: new ops code auto-converts. If correct then remove this test.
    # Mismatched array float types
    # with pytest.raises(ValueError):
    #     rpj.st_fma(f.st_f32(), f.st_f64(), f.st_f32())

    # Unsupported array types
    with pytest.raises(ValueError):
        rpj.st_fma(f.st_i32(), f.st_i32(), f.st_i32())


def test_shuffle_dense_st_mul_array_mismatch(rpj_test_fixture_type_mismatch):
    f = rpj_test_fixture_type_mismatch

    # Mismatch first and second widths
    with pytest.raises(ValueError):
        rpj.st_mul(f.st_f32(2, 2), f.st_f32(3, 2))

    # FIXME for review: new ops code auto-converts. If correct then remove this test.
    # Mismatched array float types
    # with pytest.raises(ValueError):
    #     rpj.st_mul(f.st_f32(), f.st_f64())

    # Unsupported array types
    with pytest.raises(ValueError):
        rpj.st_mul(f.st_i32(), f.st_i32())


def test_shuffle_product_commutative(rpj_dtype, rpj_batch):
    basis = rpj.TensorBasis(4, 3)
    lhs = rpj_batch.rng_shuffle_tensor(basis, rpj_dtype)
    rhs = rpj_batch.rng_shuffle_tensor(basis, rpj_dtype)

    result1 = rpj.st_mul(lhs, rhs)
    result2 = rpj.st_mul(rhs, lhs)

    atol = 1e-7 if rpj_dtype == jnp.float32 else 1e-16
    assert jnp.allclose(result1.data, result2.data, atol=atol)


def test_shuffle_product_unit(rpj_dtype, rpj_batch):
    basis = rpj.TensorBasis(4, 3)
    lhs = rpj_batch.rng_shuffle_tensor(basis, rpj_dtype)

    unit_data = np.zeros_like(lhs.data)
    unit_data[...,0] = 1.
    unit = rpj.ShuffleTensor(unit_data, lhs.basis)

    result1 = rpj.st_mul(lhs, unit)
    result2 = rpj.st_mul(unit, lhs)

    assert jnp.allclose(result1.data, lhs.data)
    assert jnp.allclose(result2.data, lhs.data)


def test_shuffle_product_two_letters(rpj_dtype, rpj_batch):
    basis = rpj.TensorBasis(4, 3)
    to_idx = _word_to_idx_fn(4)

    lhs_data = np.zeros(rpj_batch.tensor_batch_shape(basis), dtype=rpj_dtype)
    rhs_data = np.zeros(rpj_batch.tensor_batch_shape(basis), dtype=rpj_dtype)

    lhs_data[..., 1] = 1.0
    rhs_data[..., 2] = 1.0

    lhs = rpj.ShuffleTensor(lhs_data, basis)
    rhs = rpj.ShuffleTensor(rhs_data, basis)

    result = rpj.st_mul(lhs, rhs)

    expected = np.zeros_like(lhs.data)
    expected[..., to_idx(1, 2)] = 1.0
    expected[..., to_idx(2, 1)] = 1.0

    assert jnp.allclose(result.data, expected)


def test_shuffle_product_letter_and_deg2(rpj_dtype, rpj_batch):
    basis = rpj.TensorBasis(4, 3)
    to_idx = _word_to_idx_fn(4)

    lhs_data = np.zeros(rpj_batch.tensor_batch_shape(basis), dtype=rpj_dtype)
    rhs_data = np.zeros(rpj_batch.tensor_batch_shape(basis), dtype=rpj_dtype)

    lhs_data[..., to_idx(1, 2)] = 1.0
    rhs_data[..., 3] = 1.0

    lhs = rpj.ShuffleTensor(lhs_data, basis)
    rhs = rpj.ShuffleTensor(rhs_data, basis)

    result = rpj.st_mul(lhs, rhs)

    expected = np.zeros_like(lhs.data)
    expected[..., to_idx(1, 2, 3)] = 1.0
    expected[..., to_idx(1, 3, 2)] = 1.0
    expected[..., to_idx(3, 1, 2)] = 1.0

    assert jnp.allclose(result.data, expected)
