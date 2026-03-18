from functools import partial

import jax
import jax.numpy as jnp
import pytest
import roughpy_jax as rpj
from jax import test_util as jtu

from derivative_testing import (
    DerivativeTrialsHelper,
    assert_is_adjoint_derivative,
    assert_is_derivative,
    assert_is_linear,
)
from roughpy_jax.algebra import (
    st_adjoint_mul,
    st_adjoint_mul_adjoint_derivative,
    st_adjoint_mul_derivative,
)


@pytest.fixture(params=[jnp.float32, jnp.float64])
def shuffle_deriv_trials(request):
    yield DerivativeTrialsHelper(request.param, width=4, depth=3)


def test_shuffle_dense_st_adj_mul_array_mismatch(rpj_test_fixture_type_mismatch):
    f = rpj_test_fixture_type_mismatch

    # Mismatch op and arg widths
    with pytest.raises(ValueError):
        st_adjoint_mul(f.st_f32(2, 2), f.ft_f32(3, 2))


def test_shuffle_st_adj_mul_identity(rpj_dtype, rpj_batch):
    basis = rpj.TensorBasis(4, 3)

    op_data = rpj_batch.identity_zero_data(basis, rpj_dtype)
    op = rpj.ShuffleTensor(op_data, basis)
    arg = rpj_batch.rng_nonzero_free_tensor(basis, rpj_dtype)

    result = st_adjoint_mul(op, arg)

    assert jnp.allclose(result.data, arg.data)


def test_shuffle_st_adj_mul_random_equivalent(rpj_dtype, rpj_batch):
    basis = rpj.TensorBasis(4, 3)

    op = rpj_batch.rng_shuffle_tensor(basis, rpj_dtype)
    functional = rpj_batch.rng_shuffle_tensor(basis, rpj_dtype)
    argument = rpj_batch.rng_nonzero_free_tensor(basis, rpj_dtype)

    lhs = rpj.tensor_pairing(functional, st_adjoint_mul(op, argument))
    rhs = rpj.tensor_pairing(rpj.st_mul(op, functional), argument)

    atol = 1e-7 if rpj_dtype == jnp.float32 else 1e-16
    assert jnp.allclose(lhs, rhs, atol=atol)


def test_shuffle_st_adj_mul_linear_in_op(rpj_dtype, rpj_small_batch):
    basis = rpj.TensorBasis(4, 3)

    op_x = rpj_small_batch.rng_shuffle_tensor(basis, rpj_dtype)
    op_y = rpj_small_batch.rng_shuffle_tensor(basis, rpj_dtype)
    arg = rpj_small_batch.rng_nonzero_free_tensor(basis, rpj_dtype)
    alpha = jnp.asarray(0.7, dtype=rpj_dtype)
    beta = jnp.asarray(-1.3, dtype=rpj_dtype)

    def fn(op):
        return st_adjoint_mul(op, arg)

    assert_is_linear(fn, op_x, op_y, alpha, beta)


def test_shuffle_st_adj_mul_linear_in_arg(rpj_dtype, rpj_small_batch):
    basis = rpj.TensorBasis(4, 3)

    op = rpj_small_batch.rng_shuffle_tensor(basis, rpj_dtype)
    arg_x = rpj_small_batch.rng_nonzero_free_tensor(basis, rpj_dtype)
    arg_y = rpj_small_batch.rng_nonzero_free_tensor(basis, rpj_dtype)
    alpha = jnp.asarray(0.6, dtype=rpj_dtype)
    beta = jnp.asarray(-0.8, dtype=rpj_dtype)

    def fn(arg):
        return st_adjoint_mul(op, arg)

    assert_is_linear(fn, arg_x, arg_y, alpha, beta)


def test_st_adjoint_mul_check_vjp(rpj_batch):
    rpj_dtype = jnp.dtype("float32")
    basis = rpj.TensorBasis(4, 3)
    op = rpj_batch.rng_shuffle_tensor(basis, rpj_dtype)
    arg = rpj_batch.rng_nonzero_free_tensor(basis, rpj_dtype)

    def adj_mul_(op, arg):
        op.data = jnp.asarray(op.data)
        arg.data = jnp.asarray(arg.data)
        return st_adjoint_mul(op, arg)

    jtu.check_vjp(
        adj_mul_,
        partial(jax.vjp, adj_mul_),
        (op, arg),
        atol=5e-2,
        rtol=5e-2,
    )


def test_st_adjoint_mul_derivative_linear_in_t_op(shuffle_deriv_trials):
    op = shuffle_deriv_trials.uniform_shuffle_tensor()
    arg = shuffle_deriv_trials.uniform_free_tensor()
    zero_t_arg = shuffle_deriv_trials.zero_free_tensor()
    t_op_x = shuffle_deriv_trials.uniform_shuffle_tensor()
    t_op_y = shuffle_deriv_trials.uniform_shuffle_tensor()
    vals = shuffle_deriv_trials.uniform_data((2,))
    alpha = float(vals[0])
    beta = float(vals[1])

    def fn(t_op):
        return st_adjoint_mul_derivative(op, arg, t_op, zero_t_arg)

    assert_is_linear(fn, t_op_x, t_op_y, alpha, beta)


def test_st_adjoint_mul_derivative_linear_in_t_arg(shuffle_deriv_trials):
    op = shuffle_deriv_trials.uniform_shuffle_tensor()
    arg = shuffle_deriv_trials.uniform_free_tensor()
    zero_t_op = shuffle_deriv_trials.zero_shuffle_tensor()
    t_arg_x = shuffle_deriv_trials.uniform_free_tensor()
    t_arg_y = shuffle_deriv_trials.uniform_free_tensor()
    vals = shuffle_deriv_trials.uniform_data((2,))
    alpha = float(vals[0])
    beta = float(vals[1])

    def fn(t_arg):
        return st_adjoint_mul_derivative(op, arg, zero_t_op, t_arg)

    assert_is_linear(fn, t_arg_x, t_arg_y, alpha, beta)


def test_st_adjoint_mul_derivative_wrt_op(shuffle_deriv_trials):
    op = shuffle_deriv_trials.uniform_shuffle_tensor()
    arg = shuffle_deriv_trials.uniform_free_tensor()
    tangent = shuffle_deriv_trials.uniform_shuffle_tensor() * shuffle_deriv_trials.cond_dtype(
        1e-3, 1e0
    )
    zero_t_arg = shuffle_deriv_trials.zero_free_tensor()

    def fn(arg_op):
        return st_adjoint_mul(arg_op, arg)

    def fn_deriv(arg_op, t_arg_op):
        return st_adjoint_mul_derivative(arg_op, arg, t_arg_op, zero_t_arg)

    assert_is_derivative(
        fn,
        fn_deriv,
        op,
        tangent,
        eps_factors=(1.0e-2, 3.0e-3, 1.0e-3),
        abs_tol=shuffle_deriv_trials.cond_dtype(5e-2, 1e-5),
        rel_tol=shuffle_deriv_trials.cond_dtype(5e-2, 1e-5),
    )


def test_st_adjoint_mul_derivative_wrt_arg(shuffle_deriv_trials):
    op = shuffle_deriv_trials.uniform_shuffle_tensor()
    arg = shuffle_deriv_trials.uniform_free_tensor()
    tangent = shuffle_deriv_trials.uniform_free_tensor() * shuffle_deriv_trials.cond_dtype(
        1e-3, 1e0
    )
    zero_t_op = shuffle_deriv_trials.zero_shuffle_tensor()

    def fn(arg_arg):
        return st_adjoint_mul(op, arg_arg)

    def fn_deriv(arg_arg, t_arg_arg):
        return st_adjoint_mul_derivative(op, arg_arg, zero_t_op, t_arg_arg)

    assert_is_derivative(
        fn,
        fn_deriv,
        arg,
        tangent,
        eps_factors=(1.0e-2, 3.0e-3, 1.0e-3),
        abs_tol=shuffle_deriv_trials.cond_dtype(5e-2, 1e-5),
        rel_tol=shuffle_deriv_trials.cond_dtype(5e-2, 1e-5),
    )


def test_st_adjoint_mul_adjoint_derivative_wrt_op(shuffle_deriv_trials):
    op = shuffle_deriv_trials.uniform_shuffle_tensor()
    arg = shuffle_deriv_trials.uniform_free_tensor()
    tangent = shuffle_deriv_trials.uniform_shuffle_tensor() * shuffle_deriv_trials.cond_dtype(
        1e-3, 1e0
    )
    cotangent = shuffle_deriv_trials.uniform_shuffle_tensor()

    def fn(arg_op):
        return st_adjoint_mul(arg_op, arg)

    def fn_adj_deriv(arg_op, ct_result):
        return st_adjoint_mul_adjoint_derivative(arg_op, arg, ct_result)[0]

    assert_is_adjoint_derivative(
        fn,
        fn_adj_deriv,
        op,
        tangent,
        cotangent,
        domain_pairing=rpj.tensor_pairing,
        codomain_pairing=rpj.tensor_pairing,
        eps_factors=(1.0e-2, 3.0e-3, 1.0e-3),
        abs_tol=shuffle_deriv_trials.cond_dtype(5e-2, 1e-5),
        rel_tol=shuffle_deriv_trials.cond_dtype(5e-2, 1e-5),
    )


def test_st_adjoint_mul_adjoint_derivative_wrt_arg(shuffle_deriv_trials):
    op = shuffle_deriv_trials.uniform_shuffle_tensor()
    arg = shuffle_deriv_trials.uniform_free_tensor()
    tangent = shuffle_deriv_trials.uniform_free_tensor() * shuffle_deriv_trials.cond_dtype(
        1e-3, 1e0
    )
    cotangent = shuffle_deriv_trials.uniform_shuffle_tensor()

    def fn(arg_arg):
        return st_adjoint_mul(op, arg_arg)

    def fn_adj_deriv(arg_arg, ct_result):
        return st_adjoint_mul_adjoint_derivative(op, arg_arg, ct_result)[1]

    assert_is_adjoint_derivative(
        fn,
        fn_adj_deriv,
        arg,
        tangent,
        cotangent,
        domain_pairing=rpj.tensor_pairing,
        codomain_pairing=rpj.tensor_pairing,
        eps_factors=(1.0e-2, 3.0e-3, 1.0e-3),
        abs_tol=shuffle_deriv_trials.cond_dtype(5e-2, 1e-5),
        rel_tol=shuffle_deriv_trials.cond_dtype(5e-2, 1e-5),
    )
