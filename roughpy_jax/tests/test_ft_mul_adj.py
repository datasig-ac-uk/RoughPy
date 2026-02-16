import jax
import jax.numpy as jnp
import numpy as np
import roughpy_jax as rpj


def _random_shuffle_tensor(rng, basis, dtype, shape):
    s_data = jax.random.uniform(
        rng,
        minval=-1.0,
        maxval=1.0,
        dtype=dtype,
        shape=shape
    )
    s = rpj.DenseShuffleTensor(s_data, basis)
    return s


def test_adjoint_ft_mul_identity(rpj_dtype, rpj_batch, rpj_no_acceleration):
    rng = jax.random.key(12345)
    basis = rpj.TensorBasis(2, 2)

    a_data = rpj_batch.repeat(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], rpj_dtype))
    a = rpj.DenseFreeTensor(a_data, basis)

    s = _random_shuffle_tensor(rng, basis, rpj_dtype, rpj_batch.tensor_batch_shape(basis))

    lmul = rpj.ft_adjoint_left_mul(a, s)
    assert jnp.allclose(lmul.data, s.data)

    rmul = rpj.ft_adjoint_right_mul(a, s)

    # Right multiply switches elements 4 and 5 and flips sign of elements 1 and 2
    perm = jnp.array([0, 1, 2, 3, 5, 4, 6])
    sign = jnp.array([1, -1, -1, 1, 1, 1, 1], dtype=rpj_dtype)
    expected_rmul = s.data[..., perm] * sign

    assert jnp.allclose(rmul.data, expected_rmul)


def test_adjoint_ft_mul_letter(rpj_dtype, rpj_batch, rpj_no_acceleration):
    rng = jax.random.key(12345)

    basis = rpj.TensorBasis(2, 2)
    a_data = rpj_batch.repeat(jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], rpj_dtype))
    a = rpj.DenseFreeTensor(a_data, basis)

    s = _random_shuffle_tensor(rng, basis, rpj_dtype, rpj_batch.tensor_batch_shape(basis))

    lmul = rpj.ft_adjoint_left_mul(a, s)

    # Expected values are values from index 1, 3 and 4 going into 0, 1, 2, i.e. for non-batched
    # data, expected_data = jnp.array([s.data[1], s.data[3], s.data[4], 0.0, 0.0, 0.0, 0.0])
    s_indexes = jnp.array([1, 3, 4])
    expected_lmul = jnp.zeros_like(s.data).at[..., :3].set(s.data[..., s_indexes])

    assert jnp.allclose(lmul.data, expected_lmul)

    rmul = rpj.ft_adjoint_right_mul(a, s)

    # Right multiply flips sign of element 0
    sign = jnp.array([-1, 1, 1, 1, 1, 1, 1], dtype=rpj_dtype)
    expected_rmul = expected_lmul * sign
    assert jnp.allclose(rmul.data, expected_rmul)


def test_adjoint_ft_mul_random_equivalent(rpj_dtype, rpj_batch, rpj_no_acceleration):
    rng = jax.random.key(12345)
    basis = rpj.TensorBasis(2, 2)

    a_data = jnp.zeros((basis.size(),), dtype=jnp.float32)
    a_data = a_data.at[1:basis.width + 1].set(jax.random.normal(rng, shape=(basis.width,), dtype=jnp.float32))
    a = rpj.DenseFreeTensor(a_data, basis)

    s = _random_shuffle_tensor(rng, basis, jnp.float32, (basis.size(),))

    lmul = rpj.ft_adjoint_left_mul(a, s)
    rmul = rpj.ft_adjoint_right_mul(a, s)

    # Right multiply flips sign of element 0
    sign = jnp.array([-1, 1, 1, 1, 1, 1, 1], dtype=rpj_dtype)
    rmul_expected = lmul.data * sign
    assert jnp.allclose(rmul.data, rmul_expected)
