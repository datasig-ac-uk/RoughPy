import pytest
from derivative_testing import (
    assert_is_adjoint,
    assert_is_derivative,
    assert_is_adjoint_derivative,
    assert_is_linear,
)

import numpy as np


def test_assert_is_linear():
    rng = np.random.default_rng(12345)

    x = rng.uniform(-1.0, 1.0, size=(2,))
    y = rng.uniform(-1.0, 1.0, size=(2,))

    alpha = rng.uniform(-1.0, 1.0)
    beta = rng.uniform(-1.0, 1.0)

    def linear_fn(x):
        return 2.0 * x

    assert_is_linear(linear_fn, x, y, alpha, beta)

    def nonlinear_fn(x):
        return x * x

    with pytest.raises(AssertionError):
        assert_is_linear(nonlinear_fn, x, y, alpha, beta)


def test_assert_is_derivative():
    rng = np.random.default_rng(12345)

    x = rng.uniform(-1.0, 1.0)
    h = rng.uniform(-1.0, 1.0)

    def fn(x):
        return x * x

    def fn_deriv(x, h):
        return 2.0 * x * h

    assert_is_derivative(fn, fn_deriv, x, h)


def test_assert_is_adjoint():
    rng = np.random.default_rng(12345)

    x = rng.uniform(-1.0, 1.0, size=(3,))
    functional = rng.uniform(-1.0, 1.0, size=(2,))
    mat = rng.uniform(-1.0, 1.0, size=(2, 3))

    def fn(x):
        return mat @ x

    def fn_adj(functional):
        return mat.T @ functional

    def pairing(lhs, rhs):
        return np.dot(lhs, rhs)

    assert_is_adjoint(fn, fn_adj, x, functional, pairing, pairing)


def test_assert_is_adjoint_fails_for_bad_adjoint():
    rng = np.random.default_rng(12345)

    x = rng.uniform(-1.0, 1.0, size=(3,))
    functional = rng.uniform(-1.0, 1.0, size=(2,))
    mat = rng.uniform(-1.0, 1.0, size=(2, 3))

    def fn(x):
        return mat @ x

    def bad_fn_adj(functional):
        return 1.5 * (mat.T @ functional)

    def pairing(lhs, rhs):
        return np.dot(lhs, rhs)

    with pytest.raises(AssertionError):
        assert_is_adjoint(fn, bad_fn_adj, x, functional, pairing, pairing)


def test_assert_is_adjoint_derivative():
    rng = np.random.default_rng(12345)

    x = rng.uniform(-1.0, 1.0, size=(3,))
    tangent = rng.uniform(-1.0, 1.0, size=(3,))
    cotangent = rng.uniform(-1.0, 1.0, size=(3,))

    def fn(x):
        return x * x

    def fn_adj_deriv(x, cotangent):
        return 2.0 * x * cotangent

    def pairing(lhs, rhs):
        return np.dot(lhs, rhs)

    assert_is_adjoint_derivative(
        fn, fn_adj_deriv, x, tangent, cotangent, pairing, pairing
    )


def test_assert_is_adjoint_derivative_fails_for_bad_adjoint():
    rng = np.random.default_rng(12345)

    x = rng.uniform(-1.0, 1.0, size=(3,))
    tangent = rng.uniform(-1.0, 1.0, size=(3,))
    cotangent = rng.uniform(-1.0, 1.0, size=(3,))
    mat = rng.uniform(-1.0, 1.0, size=(3, 3))

    def fn(x):
        return mat @ x

    def bad_fn_adj_deriv(x, cotangent):
        return 1.5 * (mat.T @ cotangent)

    def pairing(lhs, rhs):
        return np.dot(lhs, rhs)

    with pytest.raises(AssertionError):
        assert_is_adjoint_derivative(
            fn, bad_fn_adj_deriv, x, tangent, cotangent, pairing, pairing
        )
