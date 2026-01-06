import jax.numpy as jnp
import jax.random
import numpy as np
import pytest
import roughpy_jax as rpj

from rpy_test_common import array_dtypes, jnp_to_np_float


def _create_non_zero_ft(rng, basis, jnp_dtype):
    """
    The tests below need to operate on non-zero values otherwise exp/log will
    end up always being zero and tests are not doing anything useful. Only the
    vector part (directly after scalar part, hence [1:width+1]) needs to be set
    to ensure the overall value does not collapse to zero.
    """
    data = np.zeros(basis.size(), dtype=jnp_to_np_float(jnp_dtype))
    data[1:basis.width + 1] = rng.normal(size=(basis.width,))
    return rpj.FreeTensor(data, basis)


def test_dense_ft_exp_array_mismatch(rpj_test_fixture_type_mismatch):
    f = rpj_test_fixture_type_mismatch

    # Unsupported array types
    with pytest.raises(ValueError):
        rpj.ft_exp(f.ft_i32())

    with pytest.raises(ValueError):
        rpj.ft_log(f.ft_i32())


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
    a = _create_non_zero_ft(rng, basis, jnp_dtype)

    exp_a = rpj.ft_exp(a)
    log_exp_a = rpj.ft_log(exp_a)

    assert jnp.allclose(log_exp_a.data, a.data)


@pytest.mark.parametrize("jnp_dtype", array_dtypes)
def test_dense_ft_exp_and_fmexp(jnp_dtype):
    rng = np.random.default_rng()

    basis = rpj.TensorBasis(2, 2)
    e = rpj.FreeTensor(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp_dtype), basis)
    x = _create_non_zero_ft(rng, basis, jnp_dtype)

    e_exp_b = rpj.ft_fmexp(e, x)
    exp_b = rpj.ft_exp(x)
    assert jnp.allclose(e_exp_b.data, exp_b.data)


@pytest.mark.parametrize("jnp_dtype", array_dtypes)
def test_dense_ft_fmexp(jnp_dtype):
    rng = np.random.default_rng()

    basis = rpj.TensorBasis(2, 2)
    a = rpj.FreeTensor(jnp.array([1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=jnp_dtype), basis)
    x = _create_non_zero_ft(rng, basis, jnp_dtype)
    b = rpj.ft_fmexp(a, x)

    expected = rpj.ft_mul(a, rpj.ft_exp(x))
    assert jnp.allclose(b.data, expected.data)



def test_dense_ft_fmexp_batched(batch_shape):
    basis = rpj.TensorBasis(3, 3)
    rng_key = jax.random.key(12345)

    x_data = jnp.array(jax.random.uniform(rng_key, shape=(*batch_shape, basis.size())))
    a_data = jnp.array(jax.random.uniform(rng_key, shape=(*batch_shape, basis.size())))

    x = rpj.FreeTensor(x_data, basis)
    a = rpj.FreeTensor(a_data, basis)

    result = rpj.ft_fmexp(a, x)
    assert result.data.shape[:-1] == batch_shape

    expected = rpj.ft_mul(a, rpj.ft_exp(x))
    assert expected.data.shape == result.data.shape

    assert jnp.allclose(result.data, expected.data)