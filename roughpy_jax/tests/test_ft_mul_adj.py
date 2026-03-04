import jax
import jax.numpy as jnp
import numpy as np
import pytest
import roughpy_jax as rpj

from derivative_testing import (
    DerivativeTrialsHelper,
    assert_is_adjoint_derivative,
    assert_is_derivative,
    assert_is_linear,
)


@pytest.fixture(params=[jnp.float32, jnp.float64])
def adj_mul_trials(request):
    yield DerivativeTrialsHelper(request.param, width=4, depth=4)


def _random_shuffle_tensor(rng, basis, dtype, shape):
    s_data = jax.random.uniform(
        rng,
        minval=-1.0,
        maxval=1.0,
        dtype=dtype,
        shape=shape
    )
    s = rpj.DenseShuffleTensor(s_data, basis)
    return s


def test_adjoint_ft_mul_identity(rpj_dtype, rpj_batch, rpj_no_acceleration):
    rng = jax.random.key(12345)
    basis = rpj.TensorBasis(2, 2)

    a_data = rpj_batch.repeat(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], rpj_dtype))
    a = rpj.DenseFreeTensor(a_data, basis)

    s = _random_shuffle_tensor(rng, basis, rpj_dtype, rpj_batch.tensor_batch_shape(basis))

    lmul = rpj.ft_adjoint_left_mul(a, s)
    assert jnp.allclose(lmul.data, s.data)

    rmul = rpj.ft_adjoint_right_mul(a, s)
    assert jnp.allclose(rmul.data, s.data)


def test_adjoint_ft_mul_letter(rpj_dtype, rpj_batch, rpj_no_acceleration):
    rng = jax.random.key(12345)

    basis = rpj.TensorBasis(2, 2)
    a_data = rpj_batch.repeat(jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], rpj_dtype))
    a = rpj.DenseFreeTensor(a_data, basis)

    s = _random_shuffle_tensor(rng, basis, rpj_dtype, rpj_batch.tensor_batch_shape(basis))

    # Expected values are values from index 1, 3 and 4 going into 0, 1, 2, i.e. for non-batched
    # data, expected_data = jnp.array([s.data[1], s.data[3], s.data[4], 0.0, 0.0, 0.0, 0.0])
    lmul_indexes = jnp.array([1, 3, 4])
    expected_lmul = jnp.zeros_like(s.data).at[..., :3].set(s.data[..., lmul_indexes])

    lmul = rpj.ft_adjoint_left_mul(a, s)
    assert jnp.allclose(lmul.data, expected_lmul)

    # Right multiply is similar but with indexes 1, 3, 5
    rmul_indexes = jnp.array([1, 3, 5])
    expected_rmul = jnp.zeros_like(s.data).at[..., :3].set(s.data[..., rmul_indexes])

    rmul = rpj.ft_adjoint_right_mul(a, s)
    assert jnp.allclose(rmul.data, expected_rmul)


def test_adjoint_ft_mul_random_equivalent(rpj_dtype, rpj_batch, rpj_no_acceleration):
    rng = jax.random.key(12345)
    basis = rpj.TensorBasis(2, 2)

    x_data = jax.random.normal(rng, shape=(basis.size(),), dtype=rpj_dtype)
    x = rpj.DenseFreeTensor(x_data, basis)

    y_data = jax.random.normal(rng, shape=(basis.size(),), dtype=rpj_dtype)
    y = rpj.DenseFreeTensor(y_data, basis)

    shuffle = _random_shuffle_tensor(rng, basis, rpj_dtype, (basis.size(),))

    # Pair from dot products for checking adjoint equivalence <T*(x*), y> = <x*, T(y)>
    def pair_dot_data(lhs, rhs):
        return lhs.data.dot(rhs.data)

    lmul = pair_dot_data(rpj.ft_adjoint_left_mul(x, shuffle), y)
    expected_lmul = pair_dot_data(shuffle, rpj.ft_mul(x, y))
    assert jnp.allclose(lmul, expected_lmul)

    rmul = pair_dot_data(rpj.ft_adjoint_right_mul(x, shuffle), y)
    expected_rmul = pair_dot_data(shuffle, rpj.ft_mul(x, y))
    assert jnp.allclose(rmul, expected_rmul)


def _dot_pairing(lhs, rhs):
    return jnp.sum(lhs.data * rhs.data)


def _fd_eps():
    return 1.0e-2, 3.0e-3, 1.0e-3


def test_ft_adjoint_left_mul_derivative_linear_in_t_op(adj_mul_trials):
    op = adj_mul_trials.uniform_free_tensor()
    arg = adj_mul_trials.uniform_shuffle_tensor()
    zero_t_arg = rpj.ShuffleTensor(
        jnp.zeros(adj_mul_trials.batch_shape(adj_mul_trials.tensor_basis), adj_mul_trials.dtype),
        adj_mul_trials.tensor_basis,
    )
    t_op_x = adj_mul_trials.uniform_free_tensor()
    t_op_y = adj_mul_trials.uniform_free_tensor()
    vals = adj_mul_trials.uniform_data((2,))
    alpha = float(vals[0])
    beta = float(vals[1])

    def fn(t_op):
        return rpj.ft_adjoint_left_mul_derivative(op, arg, t_op, zero_t_arg)

    assert_is_linear(fn, t_op_x, t_op_y, alpha, beta)


def test_ft_adjoint_left_mul_derivative_linear_in_t_arg(adj_mul_trials):
    op = adj_mul_trials.uniform_free_tensor()
    arg = adj_mul_trials.uniform_shuffle_tensor()
    zero_t_op = rpj.FreeTensor(
        jnp.zeros(adj_mul_trials.batch_shape(adj_mul_trials.tensor_basis), adj_mul_trials.dtype),
        adj_mul_trials.tensor_basis,
    )
    t_arg_x = adj_mul_trials.uniform_shuffle_tensor()
    t_arg_y = adj_mul_trials.uniform_shuffle_tensor()
    vals = adj_mul_trials.uniform_data((2,))
    alpha = float(vals[0])
    beta = float(vals[1])

    def fn(t_arg):
        return rpj.ft_adjoint_left_mul_derivative(op, arg, zero_t_op, t_arg)

    assert_is_linear(fn, t_arg_x, t_arg_y, alpha, beta)


def test_ft_adjoint_left_mul_derivative_wrt_op(adj_mul_trials):
    op = adj_mul_trials.uniform_free_tensor()
    arg = adj_mul_trials.uniform_shuffle_tensor()
    tangent = adj_mul_trials.uniform_free_tensor() * adj_mul_trials.cond_dtype(1e-3, 1e0)
    zero_t_arg = rpj.ShuffleTensor(
        jnp.zeros(adj_mul_trials.batch_shape(adj_mul_trials.tensor_basis), adj_mul_trials.dtype),
        adj_mul_trials.tensor_basis,
    )

    def fn(arg_op):
        return rpj.ft_adjoint_left_mul(arg_op, arg)

    def fn_deriv(arg_op, t_arg_op):
        return rpj.ft_adjoint_left_mul_derivative(arg_op, arg, t_arg_op, zero_t_arg)

    assert_is_derivative(
        fn,
        fn_deriv,
        op,
        tangent,
        eps_factors=_fd_eps(),
        abs_tol=adj_mul_trials.cond_dtype(5e-2, 1e-5),
        rel_tol=adj_mul_trials.cond_dtype(5e-2, 1e-5),
    )


def test_ft_adjoint_left_mul_derivative_wrt_arg(adj_mul_trials):
    op = adj_mul_trials.uniform_free_tensor()
    arg = adj_mul_trials.uniform_shuffle_tensor()
    tangent = adj_mul_trials.uniform_shuffle_tensor() * adj_mul_trials.cond_dtype(1e-3, 1e0)
    zero_t_op = rpj.FreeTensor(
        jnp.zeros(adj_mul_trials.batch_shape(adj_mul_trials.tensor_basis), adj_mul_trials.dtype),
        adj_mul_trials.tensor_basis,
    )

    def fn(arg_arg):
        return rpj.ft_adjoint_left_mul(op, arg_arg)

    def fn_deriv(arg_arg, t_arg_arg):
        return rpj.ft_adjoint_left_mul_derivative(op, arg_arg, zero_t_op, t_arg_arg)

    assert_is_derivative(
        fn,
        fn_deriv,
        arg,
        tangent,
        eps_factors=_fd_eps(),
        abs_tol=adj_mul_trials.cond_dtype(5e-2, 1e-5),
        rel_tol=adj_mul_trials.cond_dtype(5e-2, 1e-5),
    )


def test_ft_adjoint_left_mul_adjoint_derivative_wrt_op(adj_mul_trials):
    op = adj_mul_trials.uniform_free_tensor()
    arg = adj_mul_trials.uniform_shuffle_tensor()
    tangent = adj_mul_trials.uniform_free_tensor() * adj_mul_trials.cond_dtype(1e-3, 1e0)
    cotangent = adj_mul_trials.uniform_shuffle_tensor()

    def fn(arg_op):
        return rpj.ft_adjoint_left_mul(arg_op, arg)

    def fn_adj_deriv(arg_op, ct_result):
        return rpj.ft_adjoint_left_mul_adjoint_derivative(arg_op, arg, ct_result)[0]

    assert_is_adjoint_derivative(
        fn,
        fn_adj_deriv,
        op,
        tangent,
        cotangent,
        domain_pairing=_dot_pairing,
        codomain_pairing=_dot_pairing,
        eps_factors=_fd_eps(),
        abs_tol=adj_mul_trials.cond_dtype(5e-2, 1e-5),
        rel_tol=adj_mul_trials.cond_dtype(5e-2, 1e-5),
    )


def test_ft_adjoint_left_mul_adjoint_derivative_wrt_arg(adj_mul_trials):
    op = adj_mul_trials.uniform_free_tensor()
    arg = adj_mul_trials.uniform_shuffle_tensor()
    tangent = adj_mul_trials.uniform_shuffle_tensor() * adj_mul_trials.cond_dtype(1e-3, 1e0)
    cotangent = adj_mul_trials.uniform_shuffle_tensor()

    def fn(arg_arg):
        return rpj.ft_adjoint_left_mul(op, arg_arg)

    def fn_adj_deriv(arg_arg, ct_result):
        return rpj.ft_adjoint_left_mul_adjoint_derivative(op, arg_arg, ct_result)[1]

    assert_is_adjoint_derivative(
        fn,
        fn_adj_deriv,
        arg,
        tangent,
        cotangent,
        domain_pairing=_dot_pairing,
        codomain_pairing=_dot_pairing,
        eps_factors=_fd_eps(),
        abs_tol=adj_mul_trials.cond_dtype(5e-2, 1e-5),
        rel_tol=adj_mul_trials.cond_dtype(5e-2, 1e-5),
    )


def test_ft_adjoint_right_mul_derivative_linear_in_t_op(adj_mul_trials):
    op = adj_mul_trials.uniform_free_tensor()
    arg = adj_mul_trials.uniform_shuffle_tensor()
    zero_t_arg = rpj.ShuffleTensor(
        jnp.zeros(adj_mul_trials.batch_shape(adj_mul_trials.tensor_basis), adj_mul_trials.dtype),
        adj_mul_trials.tensor_basis,
    )
    t_op_x = adj_mul_trials.uniform_free_tensor()
    t_op_y = adj_mul_trials.uniform_free_tensor()
    vals = adj_mul_trials.uniform_data((2,))
    alpha = float(vals[0])
    beta = float(vals[1])

    def fn(t_op):
        return rpj.ft_adjoint_right_mul_derivative(op, arg, t_op, zero_t_arg)

    assert_is_linear(
        fn,
        t_op_x,
        t_op_y,
        alpha,
        beta,
        abs_tol=adj_mul_trials.cond_dtype(5e-6, 1e-6),
        rel_tol=adj_mul_trials.cond_dtype(5e-6, 1e-6),
    )


def test_ft_adjoint_right_mul_derivative_linear_in_t_arg(adj_mul_trials):
    op = adj_mul_trials.uniform_free_tensor()
    arg = adj_mul_trials.uniform_shuffle_tensor()
    zero_t_op = rpj.FreeTensor(
        jnp.zeros(adj_mul_trials.batch_shape(adj_mul_trials.tensor_basis), adj_mul_trials.dtype),
        adj_mul_trials.tensor_basis,
    )
    t_arg_x = adj_mul_trials.uniform_shuffle_tensor()
    t_arg_y = adj_mul_trials.uniform_shuffle_tensor()
    vals = adj_mul_trials.uniform_data((2,))
    alpha = float(vals[0])
    beta = float(vals[1])

    def fn(t_arg):
        return rpj.ft_adjoint_right_mul_derivative(op, arg, zero_t_op, t_arg)

    assert_is_linear(fn, t_arg_x, t_arg_y, alpha, beta)


def test_ft_adjoint_right_mul_derivative_wrt_op(adj_mul_trials):
    op = adj_mul_trials.uniform_free_tensor()
    arg = adj_mul_trials.uniform_shuffle_tensor()
    tangent = adj_mul_trials.uniform_free_tensor() * adj_mul_trials.cond_dtype(1e-3, 1e0)
    zero_t_arg = rpj.ShuffleTensor(
        jnp.zeros(adj_mul_trials.batch_shape(adj_mul_trials.tensor_basis), adj_mul_trials.dtype),
        adj_mul_trials.tensor_basis,
    )

    def fn(arg_op):
        return rpj.ft_adjoint_right_mul(arg_op, arg)

    def fn_deriv(arg_op, t_arg_op):
        return rpj.ft_adjoint_right_mul_derivative(arg_op, arg, t_arg_op, zero_t_arg)

    assert_is_derivative(
        fn,
        fn_deriv,
        op,
        tangent,
        eps_factors=_fd_eps(),
        abs_tol=adj_mul_trials.cond_dtype(5e-2, 1e-5),
        rel_tol=adj_mul_trials.cond_dtype(5e-2, 1e-5),
    )


def test_ft_adjoint_right_mul_derivative_wrt_arg(adj_mul_trials):
    op = adj_mul_trials.uniform_free_tensor()
    arg = adj_mul_trials.uniform_shuffle_tensor()
    tangent = adj_mul_trials.uniform_shuffle_tensor() * adj_mul_trials.cond_dtype(1e-3, 1e0)
    zero_t_op = rpj.FreeTensor(
        jnp.zeros(adj_mul_trials.batch_shape(adj_mul_trials.tensor_basis), adj_mul_trials.dtype),
        adj_mul_trials.tensor_basis,
    )

    def fn(arg_arg):
        return rpj.ft_adjoint_right_mul(op, arg_arg)

    def fn_deriv(arg_arg, t_arg_arg):
        return rpj.ft_adjoint_right_mul_derivative(op, arg_arg, zero_t_op, t_arg_arg)

    assert_is_derivative(
        fn,
        fn_deriv,
        arg,
        tangent,
        eps_factors=_fd_eps(),
        abs_tol=adj_mul_trials.cond_dtype(5e-2, 1e-5),
        rel_tol=adj_mul_trials.cond_dtype(5e-2, 1e-5),
    )


def test_ft_adjoint_right_mul_adjoint_derivative_wrt_op(adj_mul_trials):
    op = adj_mul_trials.uniform_free_tensor()
    arg = adj_mul_trials.uniform_shuffle_tensor()
    tangent = adj_mul_trials.uniform_free_tensor() * adj_mul_trials.cond_dtype(1e-3, 1e0)
    cotangent = adj_mul_trials.uniform_shuffle_tensor()

    def fn(arg_op):
        return rpj.ft_adjoint_right_mul(arg_op, arg)

    def fn_adj_deriv(arg_op, ct_result):
        return rpj.ft_adjoint_right_mul_adjoint_derivative(arg_op, arg, ct_result)[0]

    assert_is_adjoint_derivative(
        fn,
        fn_adj_deriv,
        op,
        tangent,
        cotangent,
        domain_pairing=_dot_pairing,
        codomain_pairing=_dot_pairing,
        eps_factors=_fd_eps(),
        abs_tol=adj_mul_trials.cond_dtype(5e-2, 1e-5),
        rel_tol=adj_mul_trials.cond_dtype(5e-2, 1e-5),
    )


def test_ft_adjoint_right_mul_adjoint_derivative_wrt_arg(adj_mul_trials):
    op = adj_mul_trials.uniform_free_tensor()
    arg = adj_mul_trials.uniform_shuffle_tensor()
    tangent = adj_mul_trials.uniform_shuffle_tensor() * adj_mul_trials.cond_dtype(1e-3, 1e0)
    cotangent = adj_mul_trials.uniform_shuffle_tensor()

    def fn(arg_arg):
        return rpj.ft_adjoint_right_mul(op, arg_arg)

    def fn_adj_deriv(arg_arg, ct_result):
        return rpj.ft_adjoint_right_mul_adjoint_derivative(op, arg_arg, ct_result)[1]

    assert_is_adjoint_derivative(
        fn,
        fn_adj_deriv,
        arg,
        tangent,
        cotangent,
        domain_pairing=_dot_pairing,
        codomain_pairing=_dot_pairing,
        eps_factors=_fd_eps(),
        abs_tol=adj_mul_trials.cond_dtype(5e-2, 1e-5),
        rel_tol=adj_mul_trials.cond_dtype(5e-2, 1e-5),
    )
