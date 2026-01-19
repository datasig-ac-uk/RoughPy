import jax.numpy as jnp
import numpy as np
import pytest
import roughpy_jax as rpj


def test_antipode_array_mismatch(rpj_test_fixture_type_mismatch):
    f = rpj_test_fixture_type_mismatch

    # Unsupported array types
    with pytest.raises(ValueError):
        rpj.antipode(f.ft_i32())


def test_antipode_on_signature(rpy_dtype, rpy_batch):
    """
    The antipode of a group-like free tensor (like a signature) is the
    inverse of said tensor. Thus the product of antipode(x) and x should
    be the identity.
    """
    basis = rpj.TensorBasis(4, 3)
    shape = rpy_batch.tensor_batch_shape(basis)
    x_data = np.zeros(shape, rpy_dtype)

    # Set first order block of tensor data to something so operations don't collapse to zero
    x_data[...,1:basis.width + 1] = rpy_batch.rng.normal(size=(*rpy_batch.shape, basis.width))
    x = rpj.FreeTensor(x_data, basis)

    sig = rpj.ft_exp(x)
    asig =  rpj.antipode(sig)
    product = rpj.ft_mul(asig, sig)

    # The result should be the identity, uses np not jnp for easy mutability
    expected = np.zeros(shape, rpy_dtype)
    expected[...,0] = 1.0

    atol = 1e-6 if rpy_dtype == jnp.float32 else 1e-15
    assert jnp.allclose(product.data, expected, atol=atol)


def test_antipode_idempotent(rpy_dtype, rpy_batch):
    """
    The antipode is an idempotent operation:
        if x is a free tensor then antipode(antipode(x)) == x
    This is a test that this property holds.
    """
    basis = rpj.TensorBasis(4, 3)
    shape = rpy_batch.tensor_batch_shape(basis)
    x_data = rpy_batch.rng.standard_normal(shape, rpy_dtype)

    x = rpj.FreeTensor(x_data, basis)
    ax = rpj.antipode(x)
    aax = rpj.antipode(ax)

    assert jnp.allclose(x.data, aax.data)
