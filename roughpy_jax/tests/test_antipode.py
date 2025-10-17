import jax.numpy as jnp
import numpy as np
import roughpy_jax as rpj


def test_antipode_on_signature():
    """
    The antipode of a group-like free tensor (like a signature) is the
    inverse of said tensor. Thus the product of antipode(x) and x should
    be the identity.
    """
    rng = np.random.default_rng(1234)
    width = 4
    depth = 3
    basis = rpj.TensorBasis(width, depth)

    x_data = np.zeros(basis.size(), dtype=np.float32)
    x_data[1:width + 1] = rng.normal(size=(width,))
    x = rpj.FreeTensor(x_data, basis)

    sig = rpj.ft_exp(x)
    asig =  rpj.ft_antipode(sig)
    product = rpj.ft_mul(asig, sig)

    # The result should be the identity
    expected = np.zeros_like(sig.data)
    expected[0] = 1.0

    assert jnp.allclose(product.data, expected)


def test_antipode_idempotent():
    """
    The antipode is an idempotent operation:
        if x is a free tensor then antipode(antipode(x)) == x
    This is a test that this property holds.
    """
    rng = np.random.default_rng(12345)
    width = 4
    depth = 3
    basis = rpj.TensorBasis(width, depth)

    x = rpj.FreeTensor(rng.standard_normal(size=(basis.size(),), dtype=np.float32), basis)
    ax = rpj.ft_antipode(x)
    aax = rpj.ft_antipode(ax)

    assert jnp.allclose(x.data, aax.data)
