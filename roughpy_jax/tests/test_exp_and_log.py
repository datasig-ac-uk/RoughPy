from functools import partial

import jax
import jax.numpy as jnp

from jax import test_util as jtu
from derivative_testing import (
    assert_is_adjoint_derivative,
    assert_is_derivative,
    assert_is_linear,
)
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


def test_ft_log_vjp_check_vjp(rpj_batch):
    rpj_dtype = jnp.dtype("float32")
    basis = rpj.TensorBasis(2, 2)

    x = rpj.ft_exp(rpj_batch.rng_nonzero_free_tensor(basis, rpj_dtype))
    x.data = x.data.at[..., 0].set(0)

    def log_(x):
        x.data = jnp.asarray(x.data).at[..., 0].set(0)
        return rpj.ft_log(x)

    jtu.check_vjp(log_, partial(jax.vjp, log_), (x,), atol=2e-3, rtol=2e-3)


def test_ft_exp_vjp_check_vjp(rpj_batch):
    rpj_dtype = jnp.dtype("float32")
    basis = rpj.TensorBasis(2, 2)

    x_data = jnp.asarray(rpj_batch.rng_uniform(-1.0, 1.0, basis.size(), rpj_dtype))
    x_data = x_data.at[..., 0].set(0)

    x = rpj.DenseFreeTensor(x_data, basis)

    def _exp_data(x_d):
        x_d.data = jnp.asarray(x_d.data).at[..., 0].set(0)
        return rpj.ft_exp(x_d)

    jtu.check_vjp(
        _exp_data,
        partial(jax.vjp, _exp_data),
        (x,),
        atol=2e-3,
        rtol=2e-3,
    )


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


def _ft_exp_adjoint_derivative(x, ct_result):
    return rpj.ft_exp_adjoint_derivative(x, ct_result)[0]


def test_ft_exp_derivative_linear_in_tangent(rpj_batch):
    rpj_dtype = jnp.dtype("float32")
    basis = rpj.TensorBasis(2, 4)

    x = _rng_scaled_nonzero_free_tensor(rpj_batch, basis, rpj_dtype, scale=0.2)
    t_x = _rng_scaled_nonzero_free_tensor(rpj_batch, basis, rpj_dtype, scale=0.2)
    t_y = _rng_scaled_nonzero_free_tensor(rpj_batch, basis, rpj_dtype, scale=0.2)
    alpha = jnp.asarray(0.7, rpj_dtype)
    beta = jnp.asarray(-1.3, rpj_dtype)

    fn = partial(rpj.ft_exp_derivative, x)
    assert_is_linear(fn, t_x, t_y, alpha, beta)


def test_ft_exp_derivative_satisfies_derivative_condition(rpj_batch):
    rpj_dtype = jnp.dtype("float32")
    basis = rpj.TensorBasis(2, 4)

    x = _rng_scaled_nonzero_free_tensor(rpj_batch, basis, rpj_dtype, scale=0.2)
    tangent = _rng_scaled_nonzero_free_tensor(rpj_batch, basis, rpj_dtype, scale=0.2)

    assert_is_derivative(
        rpj.ft_exp,
        rpj.ft_exp_derivative,
        x,
        tangent,
        eps_factors=(1.0e-2, 3.0e-3, 1.0e-3),
        abs_tol=5.0e-2,
        rel_tol=5.0e-2,
    )


def test_ft_exp_adjoint_derivative_linear_in_cotangent(rpj_batch):
    rpj_dtype = jnp.dtype("float32")
    basis = rpj.TensorBasis(2, 4)

    x = _rng_scaled_nonzero_free_tensor(rpj_batch, basis, rpj_dtype, scale=0.2)
    ct_x = rpj_batch.rng_shuffle_tensor(basis, rpj_dtype)
    ct_y = rpj_batch.rng_shuffle_tensor(basis, rpj_dtype)
    alpha = jnp.asarray(0.8, rpj_dtype)
    beta = jnp.asarray(-0.4, rpj_dtype)

    fn = partial(_ft_exp_adjoint_derivative, x)
    assert_is_linear(fn, ct_x, ct_y, alpha, beta)


def test_ft_exp_adjoint_derivative_satisfies_derivative_condition(rpj_batch):
    rpj_dtype = jnp.dtype("float32")
    basis = rpj.TensorBasis(2, 4)

    x = _rng_scaled_nonzero_free_tensor(rpj_batch, basis, rpj_dtype, scale=0.2)
    tangent = _rng_scaled_nonzero_free_tensor(rpj_batch, basis, rpj_dtype, scale=0.2)
    cotangent = rpj_batch.rng_shuffle_tensor(basis, rpj_dtype)

    assert_is_adjoint_derivative(
        rpj.ft_exp,
        _ft_exp_adjoint_derivative,
        x,
        tangent,
        cotangent,
        rpj.tensor_pairing,
        rpj.tensor_pairing,
        eps_factors=(1.0e-2, 3.0e-3, 1.0e-3),
        abs_tol=5.0e-2,
        rel_tol=5.0e-2,
    )


def test_ft_fmexp_custom_vjp_check_vjp(rpj_batch):
    rpj_dtype = jnp.dtype("float32")
    basis = rpj.TensorBasis(2, 4)
    multiplier = rpj.ft_exp(0.2 * rpj_batch.rng_nonzero_free_tensor(basis, rpj_dtype))

    exponent_data = jnp.asarray(
        rpj_batch.rng_uniform(-0.2, 0.2, basis.size(), rpj_dtype)
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

    atol = 5e-2
    rtol = 5e-2
    jtu.check_vjp(
        _fmexp_data,
        partial(jax.vjp, _fmexp_data),
        (multiplier, exponent),
        atol=atol,
        rtol=rtol,
    )


def _rng_exponent_tensor(rpj_batch, basis, dtype, scale=1.0):
    data = jnp.asarray(rpj_batch.rng_uniform(-scale, scale, basis.size(), dtype))
    data = data.at[..., 0].set(0)
    return rpj.FreeTensor(data, basis)


def test_ft_fmexp_derivative_linear_in_t_multiplier(rpj_batch):
    rpj_dtype = jnp.dtype("float32")
    basis = rpj.TensorBasis(2, 4)
    multiplier = rpj.ft_exp(rpj_batch.rng_nonzero_free_tensor(basis, rpj_dtype))
    exponent = _rng_exponent_tensor(rpj_batch, basis, rpj_dtype)
    zero_t_exponent = rpj.FreeTensor(rpj_batch.zeros(basis.size(), rpj_dtype), basis)

    t_mul_x = rpj_batch.rng_nonzero_free_tensor(basis, rpj_dtype)
    t_mul_y = rpj_batch.rng_nonzero_free_tensor(basis, rpj_dtype)
    alpha = jnp.asarray(0.7, rpj_dtype)
    beta = jnp.asarray(-1.3, rpj_dtype)

    def fn(t_multiplier):
        return rpj.ft_fmexp_derivative(
            multiplier, exponent, t_multiplier, zero_t_exponent
        )

    assert_is_linear(fn, t_mul_x, t_mul_y, alpha, beta)


def test_ft_fmexp_derivative_linear_in_t_exponent(rpj_batch):
    rpj_dtype = jnp.dtype("float32")
    basis = rpj.TensorBasis(2, 4)
    multiplier = rpj.ft_exp(rpj_batch.rng_nonzero_free_tensor(basis, rpj_dtype))
    exponent = _rng_exponent_tensor(rpj_batch, basis, rpj_dtype)
    zero_t_multiplier = rpj.FreeTensor(rpj_batch.zeros(basis.size(), rpj_dtype), basis)

    t_exp_x = _rng_exponent_tensor(rpj_batch, basis, rpj_dtype)
    t_exp_y = _rng_exponent_tensor(rpj_batch, basis, rpj_dtype)
    alpha = jnp.asarray(0.6, rpj_dtype)
    beta = jnp.asarray(-0.8, rpj_dtype)

    def fn(t_exponent):
        return rpj.ft_fmexp_derivative(
            multiplier, exponent, zero_t_multiplier, t_exponent
        )

    assert_is_linear(fn, t_exp_x, t_exp_y, alpha, beta)


def test_ft_fmexp_derivative_satisfies_derivative_condition(rpj_batch):
    rpj_dtype = jnp.dtype("float32")
    basis = rpj.TensorBasis(2, 4)
    multiplier = rpj.ft_exp(rpj_batch.rng_nonzero_free_tensor(basis, rpj_dtype))
    exponent = _rng_exponent_tensor(rpj_batch, basis, rpj_dtype)
    zero_t = rpj.FreeTensor(rpj_batch.zeros(basis.size(), rpj_dtype), basis)
    x_multiplier = rpj.ft_exp(rpj_batch.rng_nonzero_free_tensor(basis, rpj_dtype))
    tangent_multiplier = rpj.ft_exp(rpj_batch.rng_nonzero_free_tensor(basis, rpj_dtype))
    x_exponent = _rng_exponent_tensor(rpj_batch, basis, rpj_dtype, scale=0.2)
    tangent_exponent = _rng_exponent_tensor(rpj_batch, basis, rpj_dtype, scale=0.2)

    def fn_multiplier(arg_multiplier):
        return rpj.ft_fmexp(arg_multiplier, exponent)

    def fn_multiplier_deriv(arg_multiplier, t_arg_multiplier):
        return rpj.ft_fmexp_derivative(
            arg_multiplier, exponent, t_arg_multiplier, zero_t
        )

    def fn_exponent(arg_exponent):
        return rpj.ft_fmexp(multiplier, arg_exponent)

    def fn_exponent_deriv(arg_exponent, t_arg_exponent):
        return rpj.ft_fmexp_derivative(multiplier, arg_exponent, zero_t, t_arg_exponent)

    assert_is_derivative(
        fn_multiplier,
        fn_multiplier_deriv,
        x_multiplier,
        tangent_multiplier,
        eps_factors=(1.0e-2, 3.0e-3, 1.0e-3),
        abs_tol=3.0e-2,
        rel_tol=3.0e-2,
    )
    assert_is_derivative(
        fn_exponent,
        fn_exponent_deriv,
        x_exponent,
        tangent_exponent,
        eps_factors=(1.0e-2, 3.0e-3, 1.0e-3),
        abs_tol=3.0e-2,
        rel_tol=3.0e-2,
    )


def test_ft_fmexp_adjoint_derivative_linear_in_cotangent(rpj_batch):
    rpj_dtype = jnp.dtype("float32")
    basis = rpj.TensorBasis(2, 4)
    multiplier = rpj.ft_exp(rpj_batch.rng_nonzero_free_tensor(basis, rpj_dtype))
    exponent = _rng_exponent_tensor(rpj_batch, basis, rpj_dtype)
    ct_x = rpj_batch.rng_shuffle_tensor(basis, rpj_dtype)
    ct_y = rpj_batch.rng_shuffle_tensor(basis, rpj_dtype)
    alpha = jnp.asarray(0.8, rpj_dtype)
    beta = jnp.asarray(-0.4, rpj_dtype)

    def fn_mul(ct_result):
        return rpj.ft_fmexp_adjoint_derivative(multiplier, exponent, ct_result)[0]

    def fn_exp(ct_result):
        return rpj.ft_fmexp_adjoint_derivative(multiplier, exponent, ct_result)[1]

    assert_is_linear(fn_mul, ct_x, ct_y, alpha, beta)
    assert_is_linear(fn_exp, ct_x, ct_y, alpha, beta)


def test_ft_fmexp_adjoint_derivative_satisfies_derivative_condition(rpj_batch):
    rpj_dtype = jnp.dtype("float32")
    basis = rpj.TensorBasis(2, 4)
    multiplier = rpj.ft_exp(rpj_batch.rng_nonzero_free_tensor(basis, rpj_dtype))
    exponent = _rng_exponent_tensor(rpj_batch, basis, rpj_dtype)
    x_multiplier = rpj.ft_exp(rpj_batch.rng_nonzero_free_tensor(basis, rpj_dtype))
    tangent_multiplier = rpj.ft_exp(rpj_batch.rng_nonzero_free_tensor(basis, rpj_dtype))
    x_exponent = _rng_exponent_tensor(rpj_batch, basis, rpj_dtype, scale=0.2)
    tangent_exponent = _rng_exponent_tensor(rpj_batch, basis, rpj_dtype, scale=0.2)
    cotangent = rpj_batch.rng_shuffle_tensor(basis, rpj_dtype)

    def pairing(lhs, rhs):
        return rpj.tensor_pairing(lhs, rhs)

    def fn_multiplier(arg_multiplier):
        return rpj.ft_fmexp(arg_multiplier, exponent)

    def fn_multiplier_adj_deriv(arg_multiplier, ct_result):
        return rpj.ft_fmexp_adjoint_derivative(arg_multiplier, exponent, ct_result)[0]

    def fn_exponent(arg_exponent):
        return rpj.ft_fmexp(multiplier, arg_exponent)

    def fn_exponent_adj_deriv(arg_exponent, ct_result):
        return rpj.ft_fmexp_adjoint_derivative(multiplier, arg_exponent, ct_result)[1]

    assert_is_adjoint_derivative(
        fn_multiplier,
        fn_multiplier_adj_deriv,
        x_multiplier,
        tangent_multiplier,
        cotangent,
        pairing,
        pairing,
        eps_factors=(1.0e-2, 3.0e-3, 1.0e-3),
        abs_tol=5.0e-2,
        rel_tol=5.0e-2,
    )
    assert_is_adjoint_derivative(
        fn_exponent,
        fn_exponent_adj_deriv,
        x_exponent,
        tangent_exponent,
        cotangent,
        pairing,
        pairing,
        eps_factors=(1.0e-2, 3.0e-3, 1.0e-3),
        abs_tol=5.0e-2,
        rel_tol=5.0e-2,
    )


def _rng_scaled_nonzero_free_tensor(rpj_batch, basis, dtype, scale=1.0):
    return scale * rpj_batch.rng_nonzero_free_tensor(basis, dtype)


def _rng_scaled_group_like_tensor(rpj_batch, basis, dtype, scale=1.0):
    return rpj.ft_exp(_rng_scaled_nonzero_free_tensor(rpj_batch, basis, dtype, scale))


def test_ft_log_derivative_linear_in_tangent(rpj_batch):
    rpj_dtype = jnp.dtype("float32")
    basis = rpj.TensorBasis(2, 4)

    x = _rng_scaled_group_like_tensor(rpj_batch, basis, rpj_dtype, scale=0.2)
    t_x = _rng_scaled_nonzero_free_tensor(rpj_batch, basis, rpj_dtype, scale=0.2)
    t_y = _rng_scaled_nonzero_free_tensor(rpj_batch, basis, rpj_dtype, scale=0.2)
    alpha = jnp.asarray(0.7, rpj_dtype)
    beta = jnp.asarray(-1.3, rpj_dtype)

    fn = partial(rpj.ft_log_derivative, x)
    assert_is_linear(fn, t_x, t_y, alpha, beta)


def test_ft_log_derivative_satisfies_derivative_condition(rpj_batch):
    rpj_dtype = jnp.dtype("float32")
    basis = rpj.TensorBasis(2, 4)

    x = _rng_scaled_group_like_tensor(rpj_batch, basis, rpj_dtype, scale=0.2)
    tangent = _rng_scaled_nonzero_free_tensor(rpj_batch, basis, rpj_dtype, scale=0.2)

    fn = lambda arg: rpj.ft_log(arg).data
    fn_deriv = lambda arg, t_arg: rpj.ft_log_derivative(arg, t_arg).data

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

    fn = lambda ct_result: rpj.ft_log_adjoint_derivative(x, ct_result)[0]
    assert_is_linear(fn, ct_x, ct_y, alpha, beta)


def test_ft_log_adjoint_derivative_satisfies_derivative_condition(rpj_batch):
    rpj_dtype = jnp.dtype("float32")
    basis = rpj.TensorBasis(2, 4)

    x = _rng_scaled_group_like_tensor(rpj_batch, basis, rpj_dtype, scale=0.2)
    tangent = _rng_scaled_nonzero_free_tensor(rpj_batch, basis, rpj_dtype, scale=0.2)
    cotangent = rpj_batch.rng_shuffle_tensor(basis, rpj_dtype)

    assert_is_adjoint_derivative(
        rpj.ft_log,
        lambda arg, ct_result: rpj.ft_log_adjoint_derivative(arg, ct_result)[0],
        x,
        tangent,
        cotangent,
        rpj.tensor_pairing,
        rpj.tensor_pairing,
        eps_factors=(1.0e-2, 3.0e-3, 1.0e-3),
        abs_tol=5.0e-2,
        rel_tol=5.0e-2,
    )
