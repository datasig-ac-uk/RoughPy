import jax.numpy as jnp
import pytest
import roughpy_jax as rpj

from derivative_testing import (
    DerivativeTrialsHelper,
    assert_is_adjoint_derivative,
    assert_is_derivative,
)
from roughpy_jax.algebra import st_fma_adjoint_derivative, st_fma_derivative


@pytest.fixture(params=[jnp.float32, jnp.float64])
def shuffle_deriv_trials(request):
    yield DerivativeTrialsHelper(request.param, width=4, depth=3)


def _zero_shuffle(trials):
    return rpj.ShuffleTensor(
        jnp.zeros(trials.batch_shape(trials.tensor_basis), trials.dtype),
        trials.tensor_basis,
    )


def test_shuffle_dense_st_fma_array_mismatch(rpj_test_fixture_type_mismatch):
    f = rpj_test_fixture_type_mismatch

    # Mismatch first and second widths
    with pytest.raises(ValueError):
        rpj.st_fma(f.st_f32(2, 2), f.st_f32(3, 2), f.st_f32(2, 2))

    # Mismatch first and third widths
    with pytest.raises(ValueError):
        rpj.st_fma(f.st_f32(2, 2), f.st_f32(2, 2), f.st_f32(3, 2))


def test_st_fma_derivative_wrt_a(shuffle_deriv_trials):
    a = shuffle_deriv_trials.uniform_shuffle_tensor()
    b = shuffle_deriv_trials.uniform_shuffle_tensor()
    c = shuffle_deriv_trials.uniform_shuffle_tensor()
    tangent = shuffle_deriv_trials.uniform_shuffle_tensor() * shuffle_deriv_trials.cond_dtype(
        1e-3, 1e0
    )
    zero_t = _zero_shuffle(shuffle_deriv_trials)

    def fn(arg_a):
        return rpj.st_fma(arg_a, b, c)

    def fn_deriv(arg_a, t_arg_a):
        return st_fma_derivative(arg_a, b, c, t_arg_a, zero_t, zero_t)

    assert_is_derivative(
        fn,
        fn_deriv,
        a,
        tangent,
        eps_factors=(1.0e-2, 3.0e-3, 1.0e-3),
        abs_tol=shuffle_deriv_trials.cond_dtype(5e-2, 1e-5),
        rel_tol=shuffle_deriv_trials.cond_dtype(5e-2, 1e-5),
    )


def test_st_fma_derivative_wrt_b(shuffle_deriv_trials):
    a = shuffle_deriv_trials.uniform_shuffle_tensor()
    b = shuffle_deriv_trials.uniform_shuffle_tensor()
    c = shuffle_deriv_trials.uniform_shuffle_tensor()
    tangent = shuffle_deriv_trials.uniform_shuffle_tensor() * shuffle_deriv_trials.cond_dtype(
        1e-3, 1e0
    )
    zero_t = _zero_shuffle(shuffle_deriv_trials)

    def fn(arg_b):
        return rpj.st_fma(a, arg_b, c)

    def fn_deriv(arg_b, t_arg_b):
        return st_fma_derivative(a, arg_b, c, zero_t, t_arg_b, zero_t)

    assert_is_derivative(
        fn,
        fn_deriv,
        b,
        tangent,
        eps_factors=(1.0e-2, 3.0e-3, 1.0e-3),
        abs_tol=shuffle_deriv_trials.cond_dtype(5e-2, 1e-5),
        rel_tol=shuffle_deriv_trials.cond_dtype(5e-2, 1e-5),
    )


def test_st_fma_derivative_wrt_c(shuffle_deriv_trials):
    a = shuffle_deriv_trials.uniform_shuffle_tensor()
    b = shuffle_deriv_trials.uniform_shuffle_tensor()
    c = shuffle_deriv_trials.uniform_shuffle_tensor()
    tangent = shuffle_deriv_trials.uniform_shuffle_tensor() * shuffle_deriv_trials.cond_dtype(
        1e-3, 1e0
    )
    zero_t = _zero_shuffle(shuffle_deriv_trials)

    def fn(arg_c):
        return rpj.st_fma(a, b, arg_c)

    def fn_deriv(arg_c, t_arg_c):
        return st_fma_derivative(a, b, arg_c, zero_t, zero_t, t_arg_c)

    assert_is_derivative(
        fn,
        fn_deriv,
        c,
        tangent,
        eps_factors=(1.0e-2, 3.0e-3, 1.0e-3),
        abs_tol=shuffle_deriv_trials.cond_dtype(5e-2, 1e-5),
        rel_tol=shuffle_deriv_trials.cond_dtype(5e-2, 1e-5),
    )


def test_st_fma_adjoint_derivative_wrt_a(shuffle_deriv_trials):
    a = shuffle_deriv_trials.uniform_shuffle_tensor()
    b = shuffle_deriv_trials.uniform_shuffle_tensor()
    c = shuffle_deriv_trials.uniform_shuffle_tensor()
    tangent = shuffle_deriv_trials.uniform_shuffle_tensor() * shuffle_deriv_trials.cond_dtype(
        1e-3, 1e0
    )
    cotangent = shuffle_deriv_trials.uniform_free_tensor()

    def fn(arg_a):
        return rpj.st_fma(arg_a, b, c)

    def fn_adj_deriv(arg_a, ct_result):
        return st_fma_adjoint_derivative(arg_a, b, c, ct_result)[0]

    assert_is_adjoint_derivative(
        fn,
        fn_adj_deriv,
        a,
        tangent,
        cotangent,
        domain_pairing=rpj.tensor_pairing,
        codomain_pairing=rpj.tensor_pairing,
        eps_factors=(1.0e-2, 3.0e-3, 1.0e-3),
        abs_tol=shuffle_deriv_trials.cond_dtype(5e-2, 1e-5),
        rel_tol=shuffle_deriv_trials.cond_dtype(5e-2, 1e-5),
    )


def test_st_fma_adjoint_derivative_wrt_b(shuffle_deriv_trials):
    a = shuffle_deriv_trials.uniform_shuffle_tensor()
    b = shuffle_deriv_trials.uniform_shuffle_tensor()
    c = shuffle_deriv_trials.uniform_shuffle_tensor()
    tangent = shuffle_deriv_trials.uniform_shuffle_tensor() * shuffle_deriv_trials.cond_dtype(
        1e-3, 1e0
    )
    cotangent = shuffle_deriv_trials.uniform_free_tensor()

    def fn(arg_b):
        return rpj.st_fma(a, arg_b, c)

    def fn_adj_deriv(arg_b, ct_result):
        return st_fma_adjoint_derivative(a, arg_b, c, ct_result)[1]

    assert_is_adjoint_derivative(
        fn,
        fn_adj_deriv,
        b,
        tangent,
        cotangent,
        domain_pairing=rpj.tensor_pairing,
        codomain_pairing=rpj.tensor_pairing,
        eps_factors=(1.0e-2, 3.0e-3, 1.0e-3),
        abs_tol=shuffle_deriv_trials.cond_dtype(5e-2, 1e-5),
        rel_tol=shuffle_deriv_trials.cond_dtype(5e-2, 1e-5),
    )


def test_st_fma_adjoint_derivative_wrt_c(shuffle_deriv_trials):
    a = shuffle_deriv_trials.uniform_shuffle_tensor()
    b = shuffle_deriv_trials.uniform_shuffle_tensor()
    c = shuffle_deriv_trials.uniform_shuffle_tensor()
    tangent = shuffle_deriv_trials.uniform_shuffle_tensor() * shuffle_deriv_trials.cond_dtype(
        1e-3, 1e0
    )
    cotangent = shuffle_deriv_trials.uniform_free_tensor()

    def fn(arg_c):
        return rpj.st_fma(a, b, arg_c)

    def fn_adj_deriv(arg_c, ct_result):
        return st_fma_adjoint_derivative(a, b, arg_c, ct_result)[2]

    assert_is_adjoint_derivative(
        fn,
        fn_adj_deriv,
        c,
        tangent,
        cotangent,
        domain_pairing=rpj.tensor_pairing,
        codomain_pairing=rpj.tensor_pairing,
        eps_factors=(1.0e-2, 3.0e-3, 1.0e-3),
        abs_tol=shuffle_deriv_trials.cond_dtype(5e-2, 1e-5),
        rel_tol=shuffle_deriv_trials.cond_dtype(5e-2, 1e-5),
    )
