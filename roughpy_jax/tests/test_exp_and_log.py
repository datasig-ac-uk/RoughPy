from functools import partial

import jax
import jax.numpy as jnp
import jax.test_util as jtu

import roughpy_jax as rpj
from derivative_testing import (
    assert_is_adjoint_derivative,
    assert_is_derivative,
    assert_is_linear,
)


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


def test_ft_log_vjp_check_vjp(rpj_batch):
    rpj_dtype = jnp.dtype("float32")
    basis = rpj.TensorBasis(2, 2)

    x = rpj.ft_exp(rpj_batch.rng_nonzero_free_tensor(basis, rpj_dtype))
    x.data = x.data.at[..., 0].set(0)

    def log_(x):
        x.data = jnp.asarray(x.data).at[..., 0].set(0)
        return rpj.ft_log(x)

    jtu.check_vjp(log_, partial(jax.vjp, log_), (x,), atol=2e-3, rtol=2e-3)


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


def _rng_scaled_nonzero_free_tensor(rpj_batch, basis, dtype, scale=1.0):
    return scale * rpj_batch.rng_nonzero_free_tensor(basis, dtype)


def _rng_scaled_group_like_tensor(rpj_batch, basis, dtype, scale=1.0):
    return rpj.ft_exp(_rng_scaled_nonzero_free_tensor(rpj_batch, basis, dtype, scale))


def _zero_unit_coeff(arg):
    out = type(arg)(jnp.asarray(arg.data), arg.basis)
    out.data = out.data.at[..., 0].set(0)
    return out


def _ft_log_shim(x):
    return rpj.ft_log(_zero_unit_coeff(x))


def _ft_log_derivative_shim(x, t_x):
    return rpj.ft_log_derivative(_zero_unit_coeff(x), t_x)


def _ft_log_adjoint_derivative_shim(x, ct_result):
    return rpj.ft_log_adjoint_derivative(_zero_unit_coeff(x), ct_result)[0]


def test_ft_log_derivative_linear_in_tangent(rpj_batch):
    rpj_dtype = jnp.dtype("float32")
    basis = rpj.TensorBasis(2, 4)

    x = _rng_scaled_group_like_tensor(rpj_batch, basis, rpj_dtype, scale=0.2)
    t_x = _rng_scaled_nonzero_free_tensor(rpj_batch, basis, rpj_dtype, scale=0.2)
    t_y = _rng_scaled_nonzero_free_tensor(rpj_batch, basis, rpj_dtype, scale=0.2)
    alpha = jnp.asarray(0.7, rpj_dtype)
    beta = jnp.asarray(-1.3, rpj_dtype)

    fn = partial(_ft_log_derivative_shim, x)
    assert_is_linear(fn, t_x, t_y, alpha, beta)


def test_ft_log_derivative_satisfies_derivative_condition(rpj_batch):
    rpj_dtype = jnp.dtype("float32")
    basis = rpj.TensorBasis(2, 4)

    x = _rng_scaled_group_like_tensor(rpj_batch, basis, rpj_dtype, scale=0.2)
    tangent = _rng_scaled_nonzero_free_tensor(rpj_batch, basis, rpj_dtype, scale=0.2)

    fn = lambda arg: _ft_log_shim(arg).data
    fn_deriv = lambda arg, t_arg: _ft_log_derivative_shim(arg, t_arg).data

    assert_is_derivative(
        fn,
        fn_deriv,
        x,
        tangent,
        eps_factors=(1.0e-2, 3.0e-3, 1.0e-3),
        abs_tol=5.0e-2,
        rel_tol=5.0e-2,
    )


def test_ft_log_adjoint_derivative_linear_in_cotangent(rpj_batch):
    rpj_dtype = jnp.dtype("float32")
    basis = rpj.TensorBasis(2, 4)

    x = _rng_scaled_group_like_tensor(rpj_batch, basis, rpj_dtype, scale=0.2)
    ct_x = rpj_batch.rng_shuffle_tensor(basis, rpj_dtype)
    ct_y = rpj_batch.rng_shuffle_tensor(basis, rpj_dtype)
    alpha = jnp.asarray(0.8, rpj_dtype)
    beta = jnp.asarray(-0.4, rpj_dtype)

    fn = partial(_ft_log_adjoint_derivative_shim, x)
    assert_is_linear(fn, ct_x, ct_y, alpha, beta)


def test_ft_log_adjoint_derivative_satisfies_derivative_condition(rpj_batch):
    rpj_dtype = jnp.dtype("float32")
    basis = rpj.TensorBasis(2, 4)

    x = _rng_scaled_group_like_tensor(rpj_batch, basis, rpj_dtype, scale=0.2)
    tangent = _rng_scaled_nonzero_free_tensor(rpj_batch, basis, rpj_dtype, scale=0.2)
    cotangent = rpj_batch.rng_shuffle_tensor(basis, rpj_dtype)

    assert_is_adjoint_derivative(
        _ft_log_shim,
        _ft_log_adjoint_derivative_shim,
        x,
        tangent,
        cotangent,
        rpj.tensor_pairing,
        rpj.tensor_pairing,
        eps_factors=(1.0e-2, 3.0e-3, 1.0e-3),
        abs_tol=5.0e-2,
        rel_tol=5.0e-2,
    )
