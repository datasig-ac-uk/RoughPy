from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import roughpy_jax as rpj
from derivative_testing import (
    DerivativeTrialsHelper,
    assert_is_adjoint_derivative,
    assert_is_derivative,
)
from jax import test_util as jtu


def test_dense_ft_fma_array_mismatch(rpj_test_fixture_type_mismatch):
    f = rpj_test_fixture_type_mismatch

    # Mismatch first and second widths
    with pytest.raises(ValueError):
        rpj.ft_fma(f.ft_f32(2, 2), f.ft_f32(3, 2), f.ft_f32(2, 2))

    # Mismatch first and third widths
    with pytest.raises(ValueError):
        rpj.ft_fma(f.ft_f32(2, 2), f.ft_f32(2, 2), f.ft_f32(3, 2))


def test_dense_ft_mul_array_mismatch(rpj_test_fixture_type_mismatch):
    f = rpj_test_fixture_type_mismatch

    # Mismatch first and second widths
    with pytest.raises(ValueError):
        rpj.ft_mul(f.ft_f32(2, 2), f.ft_f32(3, 2))


def test_dense_ft_fma(rpj_dtype, rpj_batch, rpj_no_acceleration):
    basis = rpj.TensorBasis(2, 2)
    a_data = rpj_batch.zeros(basis.size(), rpj_dtype)
    b_data = rpj_batch.repeat(jnp.array([2, 1, 3, 0.5, -1, 2, 0], dtype=rpj_dtype))
    c_data = rpj_batch.repeat(jnp.array([-1, 4, 0, 1, 1, 0, 2], dtype=rpj_dtype))

    a = rpj.FreeTensor(a_data, basis)
    b = rpj.FreeTensor(b_data, basis)
    c = rpj.FreeTensor(c_data, basis)

    d = rpj.ft_fma(a, b, c)

    expected_data = rpj_batch.repeat(
        jnp.array([-2, 7, -3, 5.5, 3, 10, 4], dtype=rpj_dtype)
    )
    assert jnp.allclose(d.data, expected_data)


def test_dense_ft_fma_construction(rpj_dtype, rpj_batch, rpj_no_acceleration):
    basis = rpj.TensorBasis(2, 2)

    def _create_rng_uniform_ft():
        data = rpj_batch.rng_uniform(-1, 1, basis.size(), rpj_dtype)
        return rpj.FreeTensor(data, basis)

    batched_a = _create_rng_uniform_ft()
    batched_b = _create_rng_uniform_ft()
    batched_c = _create_rng_uniform_ft()

    batched_d = rpj.ft_fma(batched_a, batched_b, batched_c)

    # Result d is checked iterating over batch to simplify construction of expected value
    for idx in np.ndindex(rpj_batch.shape):
        a = batched_a.data[idx]
        b = batched_b.data[idx]
        c = batched_c.data[idx]
        d = batched_d.data[idx]

        expected = np.array(a)

        # scalar term
        expected[0] += b[0] * c[0]

        # first order term
        expected[1:3] += b[0] * c[1:3] + c[0] * b[1:3]

        # second order term
        expected[3:] += b[0] * c[3:] + c[0] * b[3:] + np.outer(b[1:3], c[1:3]).flatten()

        assert jnp.allclose(d, expected)


class TestFtFmaDerivative:
    @pytest.fixture(params=[jnp.float32, jnp.float64])
    def ft_fma_trials(self, request):
        yield DerivativeTrialsHelper(request.param, width=3, depth=3)

    def test_ft_fma_check_vjp(self, ft_fma_trials):
        a = ft_fma_trials.uniform_free_tensor()
        b = ft_fma_trials.uniform_free_tensor()
        c = ft_fma_trials.uniform_free_tensor()

        def fma_(a, b, c):
            a.data = jnp.asarray(a.data)
            b.data = jnp.asarray(b.data)
            c.data = jnp.asarray(c.data)
            return rpj.ft_fma(a, b, c)

        jtu.check_vjp(
            fma_,
            partial(jax.vjp, fma_),
            (a, b, c),
            atol=5e-2,
            rtol=5e-2,
        )

    def test_ft_fma_derivative_wrt_a(self, ft_fma_trials):
        a = ft_fma_trials.uniform_free_tensor()
        b = ft_fma_trials.uniform_free_tensor()
        c = ft_fma_trials.uniform_free_tensor()
        t_a = ft_fma_trials.uniform_free_tensor() * ft_fma_trials.cond_dtype(1e-3, 1e0)
        zero_ct = ft_fma_trials.zero_free_tensor()

        def fn(arg_a):
            return rpj.ft_fma(arg_a, b, c)

        def fn_derivative(arg_a, arg_t_a):
            return rpj.ft_fma_derivative(arg_a, b, c, arg_t_a, zero_ct, zero_ct)

        assert_is_derivative(
            fn,
            fn_derivative,
            a,
            t_a,
            eps_factors=(1.0e-2, 3.0e-3, 1.0e-3),
            abs_tol=5e-2,
            rel_tol=5e-2,
        )

    def test_ft_fma_derivative_wrt_b(self, ft_fma_trials):
        a = ft_fma_trials.uniform_free_tensor()
        b = ft_fma_trials.uniform_free_tensor()
        c = ft_fma_trials.uniform_free_tensor()
        t_b = ft_fma_trials.uniform_free_tensor() * ft_fma_trials.cond_dtype(1e-3, 1e0)
        zero_ct = ft_fma_trials.zero_free_tensor()

        def fn(arg_b):
            return rpj.ft_fma(a, arg_b, c)

        def fn_derivative(arg_b, arg_t_b):
            return rpj.ft_fma_derivative(a, arg_b, c, zero_ct, arg_t_b, zero_ct)

        assert_is_derivative(
            fn,
            fn_derivative,
            b,
            t_b,
            eps_factors=(1.0e-2, 3.0e-3, 1.0e-3),
            abs_tol=5e-2,
            rel_tol=5e-2,
        )

    def test_ft_fma_derivative_wrt_c(self, ft_fma_trials):
        a = ft_fma_trials.uniform_free_tensor()
        b = ft_fma_trials.uniform_free_tensor()
        c = ft_fma_trials.uniform_free_tensor()
        t_c = ft_fma_trials.uniform_free_tensor() * ft_fma_trials.cond_dtype(1e-3, 1e0)
        zero_ct = ft_fma_trials.zero_free_tensor()

        def fn(arg_c):
            return rpj.ft_fma(a, b, arg_c)

        def fn_derivative(arg_c, arg_t_c):
            return rpj.ft_fma_derivative(a, b, arg_c, zero_ct, zero_ct, arg_t_c)

        assert_is_derivative(
            fn,
            fn_derivative,
            c,
            t_c,
            eps_factors=(1.0e-2, 3.0e-3, 1.0e-3),
            abs_tol=5e-2,
            rel_tol=5e-2,
        )

    def test_ft_fma_adjoint_derivative_wrt_a(self, ft_fma_trials):
        a = ft_fma_trials.uniform_free_tensor()
        b = ft_fma_trials.uniform_free_tensor()
        c = ft_fma_trials.uniform_free_tensor()
        tangent = ft_fma_trials.uniform_free_tensor() * ft_fma_trials.cond_dtype(
            1e-3, 1e0
        )
        cotangent = ft_fma_trials.uniform_shuffle_tensor()

        def fn(arg_a):
            return rpj.ft_fma(arg_a, b, c)

        def fn_adjoint_derivative(arg_a, ct_result):
            return rpj.ft_fma_adjoint_derivative(arg_a, b, c, ct_result)[0]

        assert_is_adjoint_derivative(
            fn,
            fn_adjoint_derivative,
            a,
            tangent,
            cotangent,
            domain_pairing=rpj.tensor_pairing,
            codomain_pairing=rpj.tensor_pairing,
            eps_factors=(1.0e-2, 3.0e-3, 1.0e-3),
            abs_tol=ft_fma_trials.cond_dtype(5e-2, 1e-6),
            rel_tol=ft_fma_trials.cond_dtype(5e-2, 1e-6),
        )

    def test_ft_fma_adjoint_derivative_wrt_b(self, ft_fma_trials):
        a = ft_fma_trials.uniform_free_tensor()
        b = ft_fma_trials.uniform_free_tensor()
        c = ft_fma_trials.uniform_free_tensor()
        tangent = ft_fma_trials.uniform_free_tensor() * ft_fma_trials.cond_dtype(
            1e-3, 1e0
        )
        cotangent = ft_fma_trials.uniform_shuffle_tensor()

        def fn(arg_b):
            return rpj.ft_fma(a, arg_b, c)

        def fn_adjoint_derivative(arg_b, ct_result):
            return rpj.ft_fma_adjoint_derivative(a, arg_b, c, ct_result)[1]

        assert_is_adjoint_derivative(
            fn,
            fn_adjoint_derivative,
            b,
            tangent,
            cotangent,
            domain_pairing=rpj.tensor_pairing,
            codomain_pairing=rpj.tensor_pairing,
            eps_factors=(1.0e-2, 3.0e-3, 1.0e-3),
            abs_tol=ft_fma_trials.cond_dtype(5e-2, 1e-6),
            rel_tol=ft_fma_trials.cond_dtype(5e-2, 1e-6),
        )

    def test_ft_fma_adjoint_derivative_wrt_c(self, ft_fma_trials):
        a = ft_fma_trials.uniform_free_tensor()
        b = ft_fma_trials.uniform_free_tensor()
        c = ft_fma_trials.uniform_free_tensor()
        tangent = ft_fma_trials.uniform_free_tensor() * ft_fma_trials.cond_dtype(
            1e-3, 1e0
        )
        cotangent = ft_fma_trials.uniform_shuffle_tensor()

        def fn(arg_c):
            return rpj.ft_fma(a, b, arg_c)

        def fn_adjoint_derivative(arg_c, ct_result):
            return rpj.ft_fma_adjoint_derivative(a, b, arg_c, ct_result)[2]

        assert_is_adjoint_derivative(
            fn,
            fn_adjoint_derivative,
            c,
            tangent,
            cotangent,
            domain_pairing=rpj.tensor_pairing,
            codomain_pairing=rpj.tensor_pairing,
            eps_factors=(1.0e-2, 3.0e-3, 1.0e-3),
            abs_tol=ft_fma_trials.cond_dtype(5e-2, 1e-6),
            rel_tol=ft_fma_trials.cond_dtype(5e-2, 1e-6),
        )
