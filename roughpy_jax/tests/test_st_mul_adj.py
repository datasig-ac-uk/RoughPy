import pytest
import jax.numpy as jnp
import roughpy_jax as rpj

from derivative_testing import assert_is_linear
from roughpy_jax.algebra import st_adjoint_mul


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
