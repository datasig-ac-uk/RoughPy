from typing import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import roughpy_jax as rpj
from jax import test_util as jtu

from derivative_testing import (
    DerivativeTrialsHelper,
    assert_is_adjoint_derivative,
    assert_is_derivative,
    assert_is_linear,
)
from roughpy_jax.algebra import st_mul_adjoint_derivative, st_mul_derivative


@pytest.fixture(params=[jnp.float32, jnp.float64])
def shuffle_deriv_trials(request):
    yield DerivativeTrialsHelper(request.param, width=4, depth=3)


def _word_to_idx_fn(width: int) -> Callable[[...], int]:
    def inner(*letters) -> int:
        idx = 0
        for letter in letters:
            idx = idx * width + letter
        return idx

    return inner


def test_shuffle_dense_st_mul_array_mismatch(rpj_test_fixture_type_mismatch):
    f = rpj_test_fixture_type_mismatch

    # Mismatch first and second widths
    with pytest.raises(ValueError):
        rpj.st_mul(f.st_f32(2, 2), f.st_f32(3, 2))


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
    unit_data[..., 0] = 1.0
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


def test_st_mul_check_vjp(rpj_batch):
    rpj_dtype = jnp.dtype("float32")
    basis = rpj.TensorBasis(4, 3)
    lhs = rpj_batch.rng_shuffle_tensor(basis, rpj_dtype)
    rhs = rpj_batch.rng_shuffle_tensor(basis, rpj_dtype)

    def mul_(lhs, rhs):
        lhs.data = jnp.asarray(lhs.data)
        rhs.data = jnp.asarray(rhs.data)
        return rpj.st_mul(lhs, rhs)

    jtu.check_vjp(
        mul_,
        partial(jax.vjp, mul_),
        (lhs, rhs),
        atol=5e-2,
        rtol=5e-2,
    )


def test_st_mul_derivative_linear_in_t_lhs(shuffle_deriv_trials):
    lhs = shuffle_deriv_trials.uniform_shuffle_tensor()
    rhs = shuffle_deriv_trials.uniform_shuffle_tensor()
    zero_t_rhs = shuffle_deriv_trials.zero_shuffle_tensor()
    t_lhs_x = shuffle_deriv_trials.uniform_shuffle_tensor()
    t_lhs_y = shuffle_deriv_trials.uniform_shuffle_tensor()
    vals = shuffle_deriv_trials.uniform_data((2,))
    alpha = float(vals[0])
    beta = float(vals[1])

    def fn(t_lhs):
        return st_mul_derivative(lhs, rhs, t_lhs, zero_t_rhs)

    assert_is_linear(fn, t_lhs_x, t_lhs_y, alpha, beta)


def test_st_mul_derivative_linear_in_t_rhs(shuffle_deriv_trials):
    lhs = shuffle_deriv_trials.uniform_shuffle_tensor()
    rhs = shuffle_deriv_trials.uniform_shuffle_tensor()
    zero_t_lhs = shuffle_deriv_trials.zero_shuffle_tensor()
    t_rhs_x = shuffle_deriv_trials.uniform_shuffle_tensor()
    t_rhs_y = shuffle_deriv_trials.uniform_shuffle_tensor()
    vals = shuffle_deriv_trials.uniform_data((2,))
    alpha = float(vals[0])
    beta = float(vals[1])

    def fn(t_rhs):
        return st_mul_derivative(lhs, rhs, zero_t_lhs, t_rhs)

    assert_is_linear(fn, t_rhs_x, t_rhs_y, alpha, beta)


def test_st_mul_derivative_wrt_lhs(shuffle_deriv_trials):
    lhs = shuffle_deriv_trials.uniform_shuffle_tensor()
    rhs = shuffle_deriv_trials.uniform_shuffle_tensor()
    tangent = shuffle_deriv_trials.uniform_shuffle_tensor() * shuffle_deriv_trials.cond_dtype(
        1e-3, 1e0
    )
    zero_t_rhs = shuffle_deriv_trials.zero_shuffle_tensor()

    def fn(arg_lhs):
        return rpj.st_mul(arg_lhs, rhs)

    def fn_deriv(arg_lhs, t_arg_lhs):
        return st_mul_derivative(arg_lhs, rhs, t_arg_lhs, zero_t_rhs)

    assert_is_derivative(
        fn,
        fn_deriv,
        lhs,
        tangent,
        eps_factors=(1.0e-2, 3.0e-3, 1.0e-3),
        abs_tol=shuffle_deriv_trials.cond_dtype(5e-2, 1e-5),
        rel_tol=shuffle_deriv_trials.cond_dtype(5e-2, 1e-5),
    )


def test_st_mul_derivative_wrt_rhs(shuffle_deriv_trials):
    lhs = shuffle_deriv_trials.uniform_shuffle_tensor()
    rhs = shuffle_deriv_trials.uniform_shuffle_tensor()
    tangent = shuffle_deriv_trials.uniform_shuffle_tensor() * shuffle_deriv_trials.cond_dtype(
        1e-3, 1e0
    )
    zero_t_lhs = shuffle_deriv_trials.zero_shuffle_tensor()

    def fn(arg_rhs):
        return rpj.st_mul(lhs, arg_rhs)

    def fn_deriv(arg_rhs, t_arg_rhs):
        return st_mul_derivative(lhs, arg_rhs, zero_t_lhs, t_arg_rhs)

    assert_is_derivative(
        fn,
        fn_deriv,
        rhs,
        tangent,
        eps_factors=(1.0e-2, 3.0e-3, 1.0e-3),
        abs_tol=shuffle_deriv_trials.cond_dtype(5e-2, 1e-5),
        rel_tol=shuffle_deriv_trials.cond_dtype(5e-2, 1e-5),
    )


def test_st_mul_adjoint_derivative_wrt_lhs(shuffle_deriv_trials):
    lhs = shuffle_deriv_trials.uniform_shuffle_tensor()
    rhs = shuffle_deriv_trials.uniform_shuffle_tensor()
    tangent = shuffle_deriv_trials.uniform_shuffle_tensor() * shuffle_deriv_trials.cond_dtype(
        1e-3, 1e0
    )
    cotangent = shuffle_deriv_trials.uniform_free_tensor()

    def fn(arg_lhs):
        return rpj.st_mul(arg_lhs, rhs)

    def fn_adj_deriv(arg_lhs, ct_result):
        return st_mul_adjoint_derivative(arg_lhs, rhs, ct_result)[0]

    assert_is_adjoint_derivative(
        fn,
        fn_adj_deriv,
        lhs,
        tangent,
        cotangent,
        domain_pairing=rpj.tensor_pairing,
        codomain_pairing=rpj.tensor_pairing,
        eps_factors=(1.0e-2, 3.0e-3, 1.0e-3),
        abs_tol=shuffle_deriv_trials.cond_dtype(5e-2, 1e-5),
        rel_tol=shuffle_deriv_trials.cond_dtype(5e-2, 1e-5),
    )


def test_st_mul_adjoint_derivative_wrt_rhs(shuffle_deriv_trials):
    lhs = shuffle_deriv_trials.uniform_shuffle_tensor()
    rhs = shuffle_deriv_trials.uniform_shuffle_tensor()
    tangent = shuffle_deriv_trials.uniform_shuffle_tensor() * shuffle_deriv_trials.cond_dtype(
        1e-3, 1e0
    )
    cotangent = shuffle_deriv_trials.uniform_free_tensor()

    def fn(arg_rhs):
        return rpj.st_mul(lhs, arg_rhs)

    def fn_adj_deriv(arg_rhs, ct_result):
        return st_mul_adjoint_derivative(lhs, arg_rhs, ct_result)[1]

    assert_is_adjoint_derivative(
        fn,
        fn_adj_deriv,
        rhs,
        tangent,
        cotangent,
        domain_pairing=rpj.tensor_pairing,
        codomain_pairing=rpj.tensor_pairing,
        eps_factors=(1.0e-2, 3.0e-3, 1.0e-3),
        abs_tol=shuffle_deriv_trials.cond_dtype(5e-2, 1e-5),
        rel_tol=shuffle_deriv_trials.cond_dtype(5e-2, 1e-5),
    )
