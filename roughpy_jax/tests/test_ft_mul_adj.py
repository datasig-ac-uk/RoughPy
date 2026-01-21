import jax
import jax.numpy as jnp
import numpy as np
import roughpy_jax as rpj


def test_adjoint_left_ft_mul_identity(rpj_dtype, rpj_batch):
    basis = rpj.TensorBasis(2, 2)
    a_data = rpj_batch.repeat(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], rpj_dtype))
    a = rpj.DenseFreeTensor(a_data, basis)

    rng = jax.random.key(12345)
    s_data = jax.random.uniform(
        rng,
        minval=-1.0,
        maxval=1.0,
        dtype=rpj_dtype,
        shape=rpj_batch.tensor_batch_shape(basis)
    )
    s = rpj.DenseShuffleTensor(s_data, basis)
    r = rpj.ft_adjoint_left_mul(a, s)

    assert jnp.allclose(r.data, s.data)


def test_adjoint_left_ft_mul_letter(rpj_dtype, rpj_batch):
    basis = rpj.TensorBasis(2, 2)
    a_data = rpj_batch.repeat(jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], rpj_dtype))
    a = rpj.DenseFreeTensor(a_data, basis)

    rng = jax.random.key(12345)
    s_data = jax.random.uniform(
        rng,
        minval=-1.0,
        maxval=1.0,
        dtype=rpj_dtype,
        shape=rpj_batch.tensor_batch_shape(basis)
    )
    s = rpj.DenseShuffleTensor(s_data, basis)
    r = rpj.ft_adjoint_left_mul(a, s)

    # Expected values are values from index 1, 3 and 4 going into 0, 1, 2, i.e. for non-batched
    # data, expected_data = jnp.array([s.data[1], s.data[3], s.data[4], 0.0, 0.0, 0.0, 0.0])
    s_indexes = jnp.array([1, 3, 4])
    expected_data = jnp.zeros_like(s.data).at[..., :3].set(s.data[..., s_indexes])

    assert jnp.allclose(r.data, expected_data)
