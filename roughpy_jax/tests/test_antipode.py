import jax.numpy as jnp
import conftest
import roughpy_jax as rpj

rpj_no_acceleration = conftest.make_no_acceleration_fixture(rpj.ops.DenseAntipode)


def test_antipode_array_mismatch(rpj_test_fixture_type_mismatch):
    f = rpj_test_fixture_type_mismatch

    # FIXME for review: new ops code auto-converts. If correct then remove this test.
    # Unsupported array types
    # with pytest.raises(ValueError):
    #     rpj.antipode(f.ft_i32())


def test_antipode_on_signature(rpj_dtype, rpj_batch, rpj_no_acceleration):
    """
    The antipode of a group-like free tensor (like a signature) is the
    inverse of said tensor. Thus the product of antipode(x) and x should
    be the identity.
    """
    basis = rpj.TensorBasis(4, 3)
    x = rpj_batch.rng_nonzero_free_tensor(basis, rpj_dtype)

    sig = rpj.ft_exp(x)
    asig =  rpj.antipode(sig)
    product = rpj.ft_mul(asig, sig)

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
    ax = rpj.antipode(x)
    aax = rpj.antipode(ax)

    assert jnp.allclose(x.data, aax.data)
