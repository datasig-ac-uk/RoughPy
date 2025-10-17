import jax.numpy as jnp
import numpy as np
import roughpy_jax as rpj


def test_dense_ft_exp_zero():
    basis = rpj.TensorBasis(2, 2)
    a = rpj.FreeTensor(jnp.zeros(basis.size(), dtype=jnp.float32), basis)

    exp_a = rpj.ft_exp(a)

    expected = np.zeros_like(a.data)
    expected[0] = 1.0
    assert jnp.allclose(exp_a.data, expected)


def test_dense_ft_exp_letter():
    basis = rpj.TensorBasis(2, 2)
    a = rpj.FreeTensor(jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), basis)

    exp_a = rpj.ft_exp(a)

    expected = jnp.array([1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0])
    assert jnp.allclose(exp_a.data, expected)


def test_dense_ft_log_identity():
    basis = rpj.TensorBasis(2, 2)
    a = rpj.FreeTensor(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), basis)

    log_a = rpj.ft_log(a)

    expected = jnp.zeros_like(a.data)
    assert jnp.allclose(log_a.data, expected)


def test_dense_ft_exp_log_roundtrip():
    rng = np.random.default_rng()

    basis = rpj.TensorBasis(2, 2)
    a_data = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    a_data[1:basis.width+1] = rng.normal(size=(basis.width,))
    a = rpj.FreeTensor(a_data, basis)

    exp_a = rpj.ft_exp(a)
    log_exp_a = rpj.ft_log(exp_a)

    assert jnp.allclose(log_exp_a.data, a_data)


def test_dense_ft_exp_and_fmexp():
    rng = np.random.default_rng()

    basis = rpj.TensorBasis(2, 2)
    e = rpj.FreeTensor(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), basis)

    x_data = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x_data[1:basis.width+1] = rng.normal(size=(basis.width,))
    x = rpj.FreeTensor(x_data, basis)

    # FIXME add ft_fmexp
    # e_exp_b = rpj.ft_fmexp(e, x)
    # exp_b = rpj.dense_ft_exp(x)
    # assert_array_almost_equal(e_exp_b.data, exp_b.data)


def test_dense_ft_fmexp():
    basis = rpj.TensorBasis(2, 2)
    a = rpj.FreeTensor(jnp.array([1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0]), basis)

    rng = jnp.random.default_rng()
    x = rpj.FreeTensor(jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), basis)
    x.data[1:basis.width+1] = rng.normal(size=(basis.width,))

    # FIXME add ft_fmexp
    # b = rpj.ft_fmexp(a, x)
    # expected = rpj.ft_mul(a, rpj.dense_ft_exp(x))
    # assert_array_almost_equal(b.data, expected.data)
