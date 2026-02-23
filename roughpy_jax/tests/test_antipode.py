import jax
import jax.numpy as jnp
import pytest
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


class AntipodeTestFixture:
    """
    Antipode test fixture for batch of w=4, d=3 tensors and n trials of random data.
    """
    def __init__(self):
        self.rng_key = jax.random.key(12345)
        self.basis = rpj.TensorBasis(4, 3)
        self.n_trials = 10

    def new_rng_key(self):
        self.rng_key, key = jax.random.split(self.rng_key)
        return key

    def batch_shape(self):
        return (self.n_trials, self.basis.size())

    def zero(self):
        data = jnp.zeros(shape=self.batch_shape, dtype=jnp.float32)
        return rpj.FreeTensor(data, self.basis)

    def uniform_data(self, shape):
        return jax.random.uniform(
            self.new_rng_key(),
            minval=-1.0,
            maxval=1.0,
            dtype=jnp.float32,
            shape=shape
        )

    def ft_uniform(self):
        return rpj.FreeTensor(
            self.uniform_data(self.batch_shape()),
            self.basis
        )


# FIXME rename
@pytest.fixture()
def fixture():
    yield AntipodeTestFixture()


def test_antipode_linear(fixture):
    x = fixture.ft_uniform()
    y = fixture.ft_uniform()

    vals = fixture.uniform_data((2,))
    alpha = float(vals[0])
    beta = float(vals[1])

    assert_is_linear(rpj.antipode, x, y, alpha, beta)


def test_antipode_derivative(fixture):
    x = fixture.ft_uniform()
    h = fixture.ft_uniform()

    def fn(x):
        return rpj.antipode(x)

    def fn_deriv(x, h):
        return rpj.antipode_derivative(x, h)

    assert_is_derivative(
        fn,
        fn_deriv,
        x,
        h,
        eps_factors=(0.001,) # FIXME fails for 0.0001 and less
    )


def test_antipode_adjoint_derivative(fixture):
    x = fixture.ft_uniform()
    tangent = fixture.ft_uniform()
    cotangent = fixture.ft_uniform()
    mat = fixture.ft_uniform()

    def fn(x):
        return rpj.ft_mul(mat, x)

    def fn_adj_deriv(x, cotangent):
        return rpj.antipode_adjoint_derivative(x, cotangent)

    def pairing(lhs, rhs):
        return jnp.sum(lhs.data * rhs.data)

    assert_is_adjoint_derivative(
        fn,
        fn_adj_deriv,
        x,
        tangent,
        cotangent,
        domain_pairing=pairing,
        codomain_pairing=pairing,
        eps_factors=(0.001,) # FIXME fix epsilons
    )
