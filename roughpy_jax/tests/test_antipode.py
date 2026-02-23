import jax
import jax.numpy as jnp
import roughpy_jax as rpj

from derivative_testing import *

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


def test_antipode_linear(rpj_dtype, rpj_batch, rpj_no_acceleration):
    basis = rpj.TensorBasis(4, 3)
    shape = rpj_batch.tensor_batch_shape(basis)
    x_data = rpj_batch.rng.standard_normal(shape, rpj_dtype)
    y_data = rpj_batch.rng.standard_normal(shape, rpj_dtype)
    x = rpj.FreeTensor(x_data, basis)
    y = rpj.FreeTensor(y_data, basis)

    n_trials = 20
    alphas = rpj_batch.rng.uniform(size=n_trials).astype(rpj_dtype)
    betas = rpj_batch.rng.uniform(size=n_trials).astype(rpj_dtype)

    for alpha, beta in zip(alphas, betas):
        assert_is_linear(rpj.antipode, x, y, alpha, beta)


def test_antipode_derivative(rpj_dtype, rpj_batch, rpj_no_acceleration):
    basis = rpj.TensorBasis(4, 3)
    shape = rpj_batch.tensor_batch_shape(basis)
    x_data = rpj_batch.rng.standard_normal(shape, rpj_dtype)
    x = rpj.FreeTensor(x_data, basis)

    # FIXME should tangent be normalised and if so how?
    tangent_data = rpj_batch.rng.standard_normal(shape, rpj_dtype)
    tangent_data_norm = np.sqrt(np.sum(tangent_data**2)) # FIXME naive implementation
    tangent = rpj.FreeTensor(tangent_data / tangent_data_norm, basis)

    n_trials = 20
    for _ in range(n_trials):
        assert_is_derivative(
            fn=rpj.antipode,
            fn_deriv=rpj.antipode_derivative,
            x=x,
            tangent=tangent,
            abs_tol=1e-2, # FIXME correct epsilon and tolerances per test's numeric type
            rel_tol=1e-2,
            eps_factors=[0.01],
        )