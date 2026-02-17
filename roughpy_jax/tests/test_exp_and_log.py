import jax.numpy as jnp
import pytest
import roughpy_jax as rpj


def test_dense_ft_exp_array_mismatch(rpj_test_fixture_type_mismatch):
    f = rpj_test_fixture_type_mismatch

    # Unsupported array types
    with pytest.raises(ValueError):
        rpj.ft_exp(f.ft_i32())

    with pytest.raises(ValueError):
        rpj.ft_log(f.ft_i32())


def test_dense_ft_exp_zero(rpj_dtype, rpj_batch):
    basis = rpj.TensorBasis(2, 2)
    data = rpj_batch.zeros(basis.size(), rpj_dtype)
    a = rpj.FreeTensor(data, basis)

    exp_a = rpj.ft_exp(a)

    expected = rpj_batch.identity_zero_data(basis, rpj_dtype)
    assert jnp.allclose(exp_a.data, expected)


def test_dense_ft_exp_letter(rpj_dtype, rpj_batch):
    basis = rpj.TensorBasis(2, 2)
    a_data = rpj_batch.repeat(jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], rpj_dtype))
    a = rpj.FreeTensor(a_data, basis)

    exp_a = rpj.ft_exp(a)

    expected = rpj_batch.repeat(jnp.array([1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0], rpj_dtype))
    assert jnp.allclose(exp_a.data, expected)


def test_dense_ft_log_identity(rpj_dtype, rpj_batch):
    basis = rpj.TensorBasis(2, 2)
    a_data = rpj_batch.repeat(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], rpj_dtype))
    a = rpj.FreeTensor(a_data, basis)

    log_a = rpj.ft_log(a)

    expected = rpj_batch.zeros(basis.size(), rpj_dtype)
    assert jnp.allclose(log_a.data, expected)


def test_dense_ft_exp_log_roundtrip(rpj_dtype, rpj_batch):
    basis = rpj.TensorBasis(2, 2)
    a = rpj_batch.rng_nonzero_free_tensor(basis, rpj_dtype)

    exp_a = rpj.ft_exp(a)
    log_exp_a = rpj.ft_log(exp_a)

    assert jnp.allclose(log_exp_a.data, a.data)


def test_dense_ft_exp_and_fmexp(rpj_dtype, rpj_batch):
    basis = rpj.TensorBasis(2, 2)
    e_data = rpj_batch.repeat(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], rpj_dtype))
    e = rpj.FreeTensor(e_data, basis)
    x = rpj_batch.rng_nonzero_free_tensor(basis, rpj_dtype)

    e_exp_b = rpj.ft_fmexp(e, x)
    exp_b = rpj.ft_exp(x)

    assert jnp.allclose(e_exp_b.data, exp_b.data)


def test_dense_ft_fmexp(rpj_dtype, rpj_batch):
    basis = rpj.TensorBasis(2, 2)
    a_data = rpj_batch.repeat(jnp.array([1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0], rpj_dtype))
    a = rpj.FreeTensor(a_data, basis)
    x = rpj_batch.rng_nonzero_free_tensor(basis, rpj_dtype)
    b = rpj.ft_fmexp(a, x)

    expected = rpj.ft_mul(a, rpj.ft_exp(x))
    assert jnp.allclose(b.data, expected.data)
