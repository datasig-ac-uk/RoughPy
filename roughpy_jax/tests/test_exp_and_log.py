from functools import partial

import jax
import jax.numpy as jnp
from jax import test_util as jtu
import roughpy_jax as rpj


def test_dense_ft_exp_zero(rpj_dtype, rpj_batch, rpj_no_acceleration):
    basis = rpj.TensorBasis(2, 2)
    data = rpj_batch.zeros(basis.size(), rpj_dtype)
    a = rpj.FreeTensor(data, basis)

    exp_a = rpj.ft_exp(a)

    expected = rpj_batch.identity_zero_data(basis, rpj_dtype)
    assert jnp.allclose(exp_a.data, expected)


def test_dense_ft_exp_letter(rpj_dtype, rpj_batch, rpj_no_acceleration):
    basis = rpj.TensorBasis(2, 2)
    a_data = rpj_batch.repeat(jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], rpj_dtype))
    a = rpj.FreeTensor(a_data, basis)

    exp_a = rpj.ft_exp(a)

    expected = rpj_batch.repeat(
        jnp.array([1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0], rpj_dtype)
    )
    assert jnp.allclose(exp_a.data, expected)


def test_dense_ft_log_identity(rpj_dtype, rpj_batch, rpj_no_acceleration):
    basis = rpj.TensorBasis(2, 2)
    a_data = rpj_batch.repeat(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], rpj_dtype))
    a = rpj.FreeTensor(a_data, basis)

    log_a = rpj.ft_log(a)

    expected = rpj_batch.zeros(basis.size(), rpj_dtype)
    assert jnp.allclose(log_a.data, expected)


def test_dense_ft_exp_log_roundtrip(rpj_dtype, rpj_batch, rpj_no_acceleration):
    basis = rpj.TensorBasis(2, 2)
    a = rpj_batch.rng_nonzero_free_tensor(basis, rpj_dtype)

    exp_a = rpj.ft_exp(a)
    log_exp_a = rpj.ft_log(exp_a)

    assert jnp.allclose(log_exp_a.data, a.data)


def test_dense_ft_exp_and_fmexp(rpj_dtype, rpj_batch, rpj_no_acceleration):
    basis = rpj.TensorBasis(2, 2)
    e_data = rpj_batch.repeat(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], rpj_dtype))
    e = rpj.FreeTensor(e_data, basis)
    x = rpj_batch.rng_nonzero_free_tensor(basis, rpj_dtype)

    e_exp_b = rpj.ft_fmexp(e, x)
    exp_b = rpj.ft_exp(x)

    assert jnp.allclose(e_exp_b.data, exp_b.data)


def test_dense_ft_fmexp(rpj_dtype, rpj_batch, rpj_no_acceleration):
    basis = rpj.TensorBasis(2, 2)
    a_data = rpj_batch.repeat(jnp.array([1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0], rpj_dtype))
    a = rpj.FreeTensor(a_data, basis)
    x = rpj_batch.rng_nonzero_free_tensor(basis, rpj_dtype)
    b = rpj.ft_fmexp(a, x)

    expected = rpj.ft_mul(a, rpj.ft_exp(x))
    assert jnp.allclose(b.data, expected.data)


def test_ft_fmexp_custom_vjp_check_vjp(rpj_batch):
    rpj_dtype = jnp.dtype("float32")
    basis = rpj.TensorBasis(2, 2)
    multiplier = rpj.ft_exp(rpj_batch.rng_nonzero_free_tensor(basis, rpj_dtype))

    exponent_data = jnp.asarray(
        rpj_batch.rng_uniform(-1.0, 1.0, basis.size(), rpj_dtype)
    )
    assert isinstance(exponent_data, jax.Array)
    exponent_data = exponent_data.at[..., 0].set(0)
    exponent = rpj.FreeTensor(exponent_data, basis)

    def _fmexp_data(multiplier_data, exponent_data):
        # The check_vjp routine below will generate cotangents and apply them
        # in a way that might walk the result off the surface of grouplike
        # elements. It might also convert these implicitly to numpy arrays,
        # causing jax-array specific calls to fail.
        multiplier_data.data = jnp.asarray(multiplier_data.data)
        exponent_data.data = jnp.asarray(exponent_data.data).at[..., 0].set(0)
        return rpj.ft_fmexp(multiplier_data, exponent_data)

    atol = 2e-3
    rtol = 2e-3
    jtu.check_vjp(
        _fmexp_data,
        partial(jax.vjp, _fmexp_data),
        (multiplier, exponent),
        atol=atol,
        rtol=rtol,
    )
