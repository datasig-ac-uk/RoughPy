import jax.numpy as jnp
import numpy as np
import pytest
import roughpy_jax as rpj

from test_common import type_mismatch_fixture, array_dtypes, jnp_to_np_float

# FIXME add mismatch tests as in test_dense_ft_fma_array_mismatch
# FIXME add JIT tests


# FIXME confirming meaning of this method in compute tests and rename
def _create_rng_zero_ft(rng, basis, jnp_dtype):
    data = np.zeros(basis.size(), dtype=jnp_to_np_float(jnp_dtype))
    data[1:basis.width + 1] = rng.normal(size=(basis.width,))
    return rpj.FreeTensor(data, basis)


@pytest.mark.parametrize("jnp_dtype", array_dtypes)
def test_dense_ft_exp_zero(jnp_dtype):
    basis = rpj.TensorBasis(2, 2)
    a = rpj.FreeTensor(jnp.zeros(basis.size(), dtype=jnp_dtype), basis)

    exp_a = rpj.ft_exp(a)

    expected = np.zeros_like(a.data)
    expected[0] = 1.0
    assert jnp.allclose(exp_a.data, expected)


@pytest.mark.parametrize("jnp_dtype", array_dtypes)
def test_dense_ft_exp_letter(jnp_dtype):
    basis = rpj.TensorBasis(2, 2)
    a = rpj.FreeTensor(jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp_dtype), basis)

    exp_a = rpj.ft_exp(a)

    expected = jnp.array([1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0])
    assert jnp.allclose(exp_a.data, expected)


@pytest.mark.parametrize("jnp_dtype", array_dtypes)
def test_dense_ft_log_identity(jnp_dtype):
    basis = rpj.TensorBasis(2, 2)
    a = rpj.FreeTensor(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp_dtype), basis)

    log_a = rpj.ft_log(a)

    expected = jnp.zeros_like(a.data)
    assert jnp.allclose(log_a.data, expected)


@pytest.mark.parametrize("jnp_dtype", array_dtypes)
def test_dense_ft_exp_log_roundtrip(jnp_dtype):
    rng = np.random.default_rng()
    basis = rpj.TensorBasis(2, 2)
    a = _create_rng_zero_ft(rng, basis, jnp_dtype)

    exp_a = rpj.ft_exp(a)
    log_exp_a = rpj.ft_log(exp_a)

    assert jnp.allclose(log_exp_a.data, a.data)


@pytest.mark.parametrize("jnp_dtype", array_dtypes)
def test_dense_ft_exp_and_fmexp(jnp_dtype):
    rng = np.random.default_rng()

    basis = rpj.TensorBasis(2, 2)
    e = rpj.FreeTensor(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp_dtype), basis)
    x = _create_rng_zero_ft(rng, basis, jnp_dtype)

    e_exp_b = rpj.ft_fmexp(e, x)
    exp_b = rpj.ft_exp(x)
    assert jnp.allclose(e_exp_b.data, exp_b.data)


@pytest.mark.parametrize("jnp_dtype", array_dtypes)
def test_dense_ft_fmexp(jnp_dtype):
    rng = np.random.default_rng()

    basis = rpj.TensorBasis(2, 2)
    a = rpj.FreeTensor(jnp.array([1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=jnp_dtype), basis)
    x = _create_rng_zero_ft(rng, basis, jnp_dtype)
    b = rpj.ft_fmexp(a, x)

    expected = rpj.ft_mul(a, rpj.ft_exp(x))
    assert jnp.allclose(b.data, expected.data)
