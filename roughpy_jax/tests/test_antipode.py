import jax
import jax.numpy as jnp
import pytest
import roughpy_jax as rpj

from derivative_testing import assert_is_linear, assert_is_derivative, assert_is_adjoint_derivative
from jax.test_util import check_vjp
from conftest import DerivativeTrialsHelper


# Antipode test fixture for batch of w=4, d=3 free tensors
@pytest.fixture(params=[jnp.float32, jnp.float64])
def trials(request):
    yield DerivativeTrialsHelper(request.param, rpj.TensorBasis(4, 3), rpj.FreeTensor)


@jax.jit
def _antipode_on_signature(x):
    sig = rpj.ft_exp(x)
    asig =  rpj.antipode(sig)
    product = rpj.ft_mul(asig, sig)
    return product


@jax.jit
def _antipode_idempotent(x):
    ax = rpj.antipode(x)
    aax = rpj.antipode(ax)
    return aax


def test_antipode_on_signature(rpj_dtype, rpj_batch, rpj_no_acceleration):
    """
    The antipode of a group-like free tensor (like a signature) is the
    inverse of said tensor. Thus the product of antipode(x) and x should
    be the identity.
    """
    basis = rpj.TensorBasis(4, 3)
    x = rpj_batch.rng_nonzero_free_tensor(basis, rpj_dtype)
    product = _antipode_on_signature(x)

    # exp * antipode results in identity
    expected = rpj_batch.identity_zero_data(basis, rpj_dtype)

    atol = 1e-6 if rpj_dtype == jnp.float32 else 1e-15
    assert jnp.allclose(product.data, expected, atol=atol)


def test_antipode_idempotent(rpj_dtype, rpj_batch, rpj_no_acceleration):
    """
    The antipode is an idempotent operation; if x is a free tensor
    then antipode(antipode(x)) == x
    """
    basis = rpj.TensorBasis(4, 3)
    shape = rpj_batch.tensor_batch_shape(basis)
    x_data = rpj_batch.rng.standard_normal(shape, rpj_dtype)

    x = rpj.FreeTensor(x_data, basis)
    aax = _antipode_idempotent(x)

    assert jnp.allclose(x.data, aax.data)


def test_antipode_linear(trials):
    x = trials.uniform_tensor()
    y = trials.uniform_tensor()

    vals = trials.uniform_data((2,))
    alpha = float(vals[0])
    beta = float(vals[1])

    assert_is_linear(rpj.antipode, x, y, alpha, beta)


def test_antipode_derivative(trials):
    x = trials.uniform_tensor()
    tangent = trials.uniform_tensor() * trials.tangent_scale

    assert_is_derivative(
        rpj.antipode,
        rpj.antipode_derivative,
        x,
        tangent,
        abs_tol=trials.base_tol
    )


def test_antipode_adjoint_derivative(trials):
    x = trials.uniform_tensor()
    tangent = trials.uniform_tensor() * trials.tangent_scale
    cotangent = trials.uniform_tensor()

    def pairing(lhs, rhs):
        return jnp.sum(lhs.data * rhs.data)

    assert_is_adjoint_derivative(
        rpj.antipode,
        rpj.antipode_adjoint_derivative,
        x,
        tangent,
        cotangent,
        domain_pairing=pairing,
        codomain_pairing=pairing,
        abs_tol=trials.base_tol * 1e1
    )


def test_antipode_pullback_vjp(trials):
    # Smoke-test VJP pullback and the bilinear identity. This complements
    # check_vjp's finite-difference check and can catch pytrees/shape/.data
    # bookkeeping or edge cases that FD sampling might miss.

    # Get cotangent returned by the VJP pullback
    x = trials.uniform_tensor()
    tangent = trials.uniform_tensor() * trials.tangent_scale
    cotangent = trials.uniform_tensor()

    y, pullback = jax.vjp(rpj.antipode, x)
    inp_cot_from_vjp = pullback(cotangent)[0]

    # Check the vjp primals are correct
    y_expected = rpj.antipode(x)
    y_dot_expected = rpj.antipode_derivative(x, tangent)
    assert jnp.allclose(y.data, y_expected.data)

    # Check bilinear identity: c·(J·u) == (J^T·c)·u
    lhs = jnp.vdot(cotangent.data, y_dot_expected.data)
    rhs = jnp.vdot(inp_cot_from_vjp.data, tangent.data)
    assert jnp.allclose(lhs, rhs)


def test_antipode_check_vjp(trials):
    # JAX built-in finite difference check
    x = trials.uniform_tensor()
    check_vjp(
        rpj.antipode,
        lambda x: jax.vjp(rpj.antipode, x),
        (x,),
        atol=trials.base_tol * 2e0,
        rtol=trials.base_tol * 2e0,
    )
