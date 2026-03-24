from functools import partial

import pytest
import jax
import jax.numpy as jnp
from jax import test_util as jtu

import roughpy_jax as rpj
from derivative_testing import (
    DerivativeTrialsHelper,
    assert_is_linear,
    assert_is_derivative,
    assert_is_adjoint_derivative,
)


def test_ft_mul(rpj_dtype, rpj_batch, rpj_no_acceleration):
    basis = rpj.TensorBasis(3, 3)

    a = rpj_batch.rng_nonzero_free_tensor(basis, rpj_dtype)
    b = rpj_batch.rng_nonzero_free_tensor(basis, rpj_dtype)

    result = rpj.ft_mul(a, b)

    z = rpj.FreeTensor.zero(basis, dtype=rpj_dtype, batch_dims=rpj_batch.shape)
    expected = rpj.ft_fma(z, a, b)

    assert jnp.allclose(result.data, expected.data)


@pytest.fixture(params=[jnp.float32, jnp.float64])
def ft_mul_trials(request):
    yield DerivativeTrialsHelper(request.param, width=3, depth=3)
    

def test_ft_mul_check_vjp(ft_mul_trials):
    lhs = ft_mul_trials.uniform_free_tensor()
    rhs = ft_mul_trials.uniform_free_tensor()

    def mul_(lhs, rhs):
        # TODO: JL - Unclear that this is necessary?
        lhs.data = jnp.asarray(lhs.data)
        rhs.data = jnp.asarray(rhs.data)
        return rpj.ft_mul(lhs, rhs)

    jtu.check_vjp(
        mul_,
        partial(jax.vjp, mul_),
        (lhs, rhs),
        atol=5e-2,
        rtol=5e-2,
    )
    
   
def test_ft_mul_derivative_linear_in_t_lhs(ft_mul_trials):
    lhs = ft_mul_trials.uniform_free_tensor()
    rhs = ft_mul_trials.uniform_free_tensor()
    zero_t_rhs = ft_mul_trials.zero_free_tensor()
    t_lhs_x = ft_mul_trials.uniform_free_tensor() * ft_mul_trials.cond_dtype(1e-3, 1e0)
    t_lhs_y = ft_mul_trials.uniform_free_tensor() * ft_mul_trials.cond_dtype(1e-3, 1e0)
    
    vals = ft_mul_trials.uniform_data((2,))
    alpha = float(vals[0])
    beta = float(vals[1])

    def fn(arg_t_lhs):
        return rpj.ft_mul_derivative(lhs, rhs, arg_t_lhs, zero_t_rhs)
        
    assert_is_linear(
        fn, 
        t_lhs_x, 
        t_lhs_y, 
        alpha, 
        beta,
    )


def test_ft_mul_derivative_linear_in_t_rhs(ft_mul_trials):
    lhs = ft_mul_trials.uniform_free_tensor()
    rhs = ft_mul_trials.uniform_free_tensor()
    zero_t_lhs = ft_mul_trials.zero_free_tensor()
    t_rhs_x = ft_mul_trials.uniform_free_tensor() * ft_mul_trials.cond_dtype(1e-3, 1e0)
    t_rhs_y = ft_mul_trials.uniform_free_tensor() * ft_mul_trials.cond_dtype(1e-3, 1e0)
    
    vals = ft_mul_trials.uniform_data((2,))
    alpha = float(vals[0])
    beta = float(vals[1])

    def fn(arg_t_rhs):
        return rpj.ft_mul_derivative(lhs, rhs, zero_t_lhs, arg_t_rhs)
        
    assert_is_linear(
        fn, 
        t_rhs_x, 
        t_rhs_y, 
        alpha, 
        beta,
    )
   
   
def test_ft_mul_derivative_wrt_rhs(ft_mul_trials):
    lhs = ft_mul_trials.uniform_free_tensor()
    rhs = ft_mul_trials.uniform_free_tensor()
    t_rhs = ft_mul_trials.uniform_free_tensor() * ft_mul_trials.cond_dtype(1e-3, 1e0)
    zero_cotangent = ft_mul_trials.zero_free_tensor()

    def fn(arg_rhs):
        return rpj.ft_mul(lhs, arg_rhs)
    
    def grad_fn(arg_rhs, t_arg_rhs):
        return rpj.ft_mul_derivative(lhs, arg_rhs, zero_cotangent, t_arg_rhs)
    
    assert_is_derivative(
        fn, 
        grad_fn, 
        rhs, 
        t_rhs, 
        eps_factors=(1.0e-2, 3.0e-3, 1.0e-3),
        abs_tol=ft_mul_trials.cond_dtype(5e-2, 1e-6),
        rel_tol=ft_mul_trials.cond_dtype(5e-2, 1e-6)
    )
    
def test_ft_mul_derivative_wrt_lhs(ft_mul_trials):
    lhs = ft_mul_trials.uniform_free_tensor()
    rhs = ft_mul_trials.uniform_free_tensor()
    t_lhs = ft_mul_trials.uniform_free_tensor() * ft_mul_trials.cond_dtype(1e-3, 1e0)
    zero_cotangent = ft_mul_trials.zero_free_tensor()

    def fn(arg_lhs):
        return rpj.ft_mul(arg_lhs, rhs)
    
    def grad_fn(arg_lhs, t_arg_lhs):
        return rpj.ft_mul_derivative(arg_lhs, rhs, t_arg_lhs, zero_cotangent)
    
    assert_is_derivative(
        fn, 
        grad_fn, 
        lhs, 
        t_lhs, 
        eps_factors=(1.0e-2, 3.0e-3, 1.0e-3),
        abs_tol=ft_mul_trials.cond_dtype(5e-2, 1e-6),
        rel_tol=ft_mul_trials.cond_dtype(5e-2, 1e-6)
    )


def test_ft_mul_adjoint_derivative_wrt_lhs(ft_mul_trials):
    lhs = ft_mul_trials.uniform_free_tensor()
    rhs = ft_mul_trials.uniform_free_tensor()
    tangent = ft_mul_trials.uniform_free_tensor() * ft_mul_trials.cond_dtype(
        1e-3, 1e0
    )
    cotangent = ft_mul_trials.uniform_free_tensor()

    def fn(arg_lhs):
        return rpj.ft_mul(arg_lhs, rhs)

    def fn_adj_deriv(arg_lhs, ct_result):
        return rpj.ft_mul_adjoint_derivative(arg_lhs, rhs, ct_result)[0]

    assert_is_adjoint_derivative(
        fn,
        fn_adj_deriv,
        lhs,
        tangent,
        cotangent,
        domain_pairing=rpj.tensor_pairing,
        codomain_pairing=rpj.tensor_pairing,
        eps_factors=(1.0e-2, 3.0e-3, 1.0e-3),
        abs_tol=ft_mul_trials.cond_dtype(5e-2, 1e-5),
        rel_tol=ft_mul_trials.cond_dtype(5e-2, 1e-5),
    )
    
def test_ft_mul_adjoint_derivative_wrt_rhs(ft_mul_trials):
    lhs = ft_mul_trials.uniform_free_tensor()
    rhs = ft_mul_trials.uniform_free_tensor()
    tangent = ft_mul_trials.uniform_free_tensor() * ft_mul_trials.cond_dtype(
        1e-3, 1e0
    )
    cotangent = ft_mul_trials.uniform_free_tensor()

    def fn(arg_rhs):
        return rpj.ft_mul(lhs, arg_rhs)

    def fn_adj_deriv(arg_rhs, ct_result):
        return rpj.ft_mul_adjoint_derivative(lhs, arg_rhs, ct_result)[1]

    assert_is_adjoint_derivative(
        fn,
        fn_adj_deriv,
        rhs,
        tangent,
        cotangent,
        domain_pairing=rpj.tensor_pairing,
        codomain_pairing=rpj.tensor_pairing,
        eps_factors=(1.0e-2, 3.0e-3, 1.0e-3),
        abs_tol=ft_mul_trials.cond_dtype(5e-2, 1e-5),
        rel_tol=ft_mul_trials.cond_dtype(5e-2, 1e-5),
    )
