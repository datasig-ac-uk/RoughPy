import jax.numpy as jnp
import numpy as np
import pytest
import roughpy_jax as rpj

from test_common import type_mismatch_fixture, array_dtypes, jnp_to_np_float

# FIXME add mismatch tests as in test_dense_ft_fma_array_mismatch
# FIXME add JIT tests

@pytest.mark.parametrize("jnp_dtype", array_dtypes)
def test_antipode_on_signature(jnp_dtype):
    """
    The antipode of a group-like free tensor (like a signature) is the
    inverse of said tensor. Thus the product of antipode(x) and x should
    be the identity.
    """
    rng = np.random.default_rng(1234)
    width = 4
    depth = 3
    basis = rpj.TensorBasis(width, depth)

    x_data = np.zeros(basis.size(), dtype=jnp_to_np_float(jnp_dtype))
    x_data[1:width + 1] = rng.normal(size=(width,))
    x = rpj.FreeTensor(x_data, basis)

    sig = rpj.ft_exp(x)
    asig =  rpj.antipode(sig)
    product = rpj.ft_mul(asig, sig)

    # The result should be the identity
    expected = np.zeros_like(sig.data)
    expected[0] = 1.0

    assert jnp.allclose(product.data, expected, atol=1e-7)


@pytest.mark.parametrize("jnp_dtype", array_dtypes)
def test_antipode_idempotent(jnp_dtype):
    """
    The antipode is an idempotent operation:
        if x is a free tensor then antipode(antipode(x)) == x
    This is a test that this property holds.
    """
    rng = np.random.default_rng(12345)
    width = 4
    depth = 3
    basis = rpj.TensorBasis(width, depth)

    x = rpj.FreeTensor(rng.standard_normal(size=(basis.size(),), dtype=jnp_to_np_float(jnp_dtype)), basis)
    ax = rpj.antipode(x)
    aax = rpj.antipode(ax)

    assert jnp.allclose(x.data, aax.data)
