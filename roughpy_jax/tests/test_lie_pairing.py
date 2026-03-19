from functools import partial

import jax
import jax.numpy as jnp
import pytest
import roughpy_jax as rpj
from derivative_testing import (
    DerivativeTrialsHelper,
    assert_is_adjoint_derivative,
    assert_is_derivative,
    assert_is_linear,
)
from jax.test_util import check_vjp


@pytest.fixture(params=[jnp.float32, jnp.float64])
def pairing_trials(request):
    yield DerivativeTrialsHelper(request.param, width=4, depth=4)


def test_dense_lie_pairing(rpj_batch):
    basis = rpj.LieBasis(2, 2)
    dtype = jnp.dtype("float32")

    functional = rpj.Lie(rpj_batch.rng_uniform(-1.0, 1.0, basis.size(), dtype), basis)
    argument = rpj.Lie(rpj_batch.rng_uniform(-1.0, 1.0, basis.size(), dtype), basis)

    result = rpj.lie_pairing(functional, argument)

    assert result.shape == rpj_batch.shape
    assert result.dtype == dtype

    expected = jnp.einsum("...i,...i -> ...", functional.data, argument.data)
    assert jnp.allclose(result, expected)


def _pairing_domain(lhs, rhs):
    return jnp.sum(lhs.data * rhs.data)


def _pairing_codomain(lhs, rhs):
    return jnp.sum(lhs * rhs)


def _fd_eps():
    return 1.0e-2, 3.0e-3, 1.0e-3


def test_lie_pairing_linear_in_functional(pairing_trials):
    argument = pairing_trials.uniform_lie()
    functional_x = pairing_trials.uniform_lie()
    functional_y = pairing_trials.uniform_lie()
    vals = pairing_trials.uniform_data((2,))
    alpha = float(vals[0])
    beta = float(vals[1])

    def fn(functional):
        return rpj.lie_pairing(functional, argument)

    assert_is_linear(
        fn,
        functional_x,
        functional_y,
        alpha,
        beta,
        abs_tol=pairing_trials.cond_dtype(5e-6, 1e-6),
        rel_tol=pairing_trials.cond_dtype(5e-6, 1e-6),
    )


def test_lie_pairing_linear_in_argument(pairing_trials):
    functional = pairing_trials.uniform_lie()
    argument_x = pairing_trials.uniform_lie()
    argument_y = pairing_trials.uniform_lie()
    vals = pairing_trials.uniform_data((2,))
    alpha = float(vals[0])
    beta = float(vals[1])

    def fn(argument):
        return rpj.lie_pairing(functional, argument)

    assert_is_linear(
        fn,
        argument_x,
        argument_y,
        alpha,
        beta,
        abs_tol=pairing_trials.cond_dtype(5e-6, 1e-6),
        rel_tol=pairing_trials.cond_dtype(5e-6, 1e-6),
    )


def test_lie_pairing_derivative_wrt_functional(pairing_trials):
    functional = pairing_trials.uniform_lie()
    argument = pairing_trials.uniform_lie()
    tangent = pairing_trials.uniform_lie() * pairing_trials.cond_dtype(1e-3, 1e0)
    zero_t_argument = rpj.Lie(
        jnp.zeros(
            pairing_trials.batch_shape(pairing_trials.lie_basis), pairing_trials.dtype
        ),
        pairing_trials.lie_basis,
    )

    def fn(arg_functional):
        return rpj.lie_pairing(arg_functional, argument)

    def fn_deriv(arg_functional, t_arg_functional):
        return rpj.lie_pairing_derivative(
            arg_functional, argument, t_arg_functional, zero_t_argument
        )

    assert_is_derivative(
        fn,
        fn_deriv,
        functional,
        tangent,
        eps_factors=_fd_eps(),
        abs_tol=pairing_trials.cond_dtype(5e-2, 1e-5),
        rel_tol=pairing_trials.cond_dtype(5e-2, 1e-5),
    )


def test_lie_pairing_derivative_wrt_argument(pairing_trials):
    functional = pairing_trials.uniform_lie()
    argument = pairing_trials.uniform_lie()
    tangent = pairing_trials.uniform_lie() * pairing_trials.cond_dtype(1e-3, 1e0)
    zero_t_functional = rpj.Lie(
        jnp.zeros(
            pairing_trials.batch_shape(pairing_trials.lie_basis), pairing_trials.dtype
        ),
        pairing_trials.lie_basis,
    )

    def fn(arg_argument):
        return rpj.lie_pairing(functional, arg_argument)

    def fn_deriv(arg_argument, t_arg_argument):
        return rpj.lie_pairing_derivative(
            functional, arg_argument, zero_t_functional, t_arg_argument
        )

    assert_is_derivative(
        fn,
        fn_deriv,
        argument,
        tangent,
        eps_factors=_fd_eps(),
        abs_tol=pairing_trials.cond_dtype(5e-2, 1e-5),
        rel_tol=pairing_trials.cond_dtype(5e-2, 1e-5),
    )


def test_lie_pairing_adjoint_derivative_wrt_functional(pairing_trials):
    functional = pairing_trials.uniform_lie()
    argument = pairing_trials.uniform_lie()
    tangent = pairing_trials.uniform_lie() * pairing_trials.cond_dtype(1e-3, 1e0)
    cotangent = pairing_trials.uniform_data((pairing_trials.n_trials,))

    def fn(arg_functional):
        return rpj.lie_pairing(arg_functional, argument)

    def fn_adj_deriv(arg_functional, ct_result):
        return rpj.lie_pairing_adjoint_derivative(arg_functional, argument, ct_result)[
            0
        ]

    assert_is_adjoint_derivative(
        fn,
        fn_adj_deriv,
        functional,
        tangent,
        cotangent,
        domain_pairing=_pairing_domain,
        codomain_pairing=_pairing_codomain,
        eps_factors=_fd_eps(),
        abs_tol=pairing_trials.cond_dtype(5e-2, 1e-5),
        rel_tol=pairing_trials.cond_dtype(5e-2, 1e-5),
    )


def test_lie_pairing_adjoint_derivative_wrt_argument(pairing_trials):
    functional = pairing_trials.uniform_lie()
    argument = pairing_trials.uniform_lie()
    tangent = pairing_trials.uniform_lie() * pairing_trials.cond_dtype(1e-3, 1e0)
    cotangent = pairing_trials.uniform_data((pairing_trials.n_trials,))

    def fn(arg_argument):
        return rpj.lie_pairing(functional, arg_argument)

    def fn_adj_deriv(arg_argument, ct_result):
        return rpj.lie_pairing_adjoint_derivative(functional, arg_argument, ct_result)[
            1
        ]

    assert_is_adjoint_derivative(
        fn,
        fn_adj_deriv,
        argument,
        tangent,
        cotangent,
        domain_pairing=_pairing_domain,
        codomain_pairing=_pairing_codomain,
        eps_factors=_fd_eps(),
        abs_tol=pairing_trials.cond_dtype(5e-2, 1e-5),
        rel_tol=pairing_trials.cond_dtype(5e-2, 1e-5),
    )


def test_lie_pairing_pullback_vjp(pairing_trials):
    functional = pairing_trials.uniform_lie()
    argument = pairing_trials.uniform_lie()
    cotangent = pairing_trials.uniform_data((pairing_trials.n_trials,))

    result, pullback = jax.vjp(rpj.lie_pairing, functional, argument)
    ct_functional, ct_argument = pullback(cotangent)

    expected = rpj.lie_pairing(functional, argument)
    expected_ct_functional, expected_ct_argument = rpj.lie_pairing_adjoint_derivative(
        functional, argument, cotangent
    )

    assert jnp.allclose(result, expected)
    assert jnp.allclose(ct_functional.data, expected_ct_functional.data)
    assert jnp.allclose(ct_argument.data, expected_ct_argument.data)


def test_lie_pairing_check_vjp(pairing_trials):
    functional = pairing_trials.uniform_lie()
    argument = pairing_trials.uniform_lie()
    base_tol = pairing_trials.cond_dtype(5e-2, 1e-5)

    check_vjp(
        rpj.lie_pairing,
        partial(jax.vjp, rpj.lie_pairing),
        (functional, argument),
        atol=base_tol,
        rtol=base_tol,
    )
