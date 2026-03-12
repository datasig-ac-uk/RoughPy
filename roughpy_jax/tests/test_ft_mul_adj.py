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
    assert jnp.allclose(rmul.data, s.data)


def test_adjoint_ft_mul_letter(rpj_dtype, rpj_batch, rpj_no_acceleration):
    rng = jax.random.key(12345)

    basis = rpj.TensorBasis(2, 2)
    a_data = rpj_batch.repeat(jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], rpj_dtype))
    a = rpj.DenseFreeTensor(a_data, basis)

    s = _random_shuffle_tensor(rng, basis, rpj_dtype, rpj_batch.tensor_batch_shape(basis))

    # Expected values are values from index 1, 3 and 4 going into 0, 1, 2, i.e. for non-batched
    # data, expected_data = jnp.array([s.data[1], s.data[3], s.data[4], 0.0, 0.0, 0.0, 0.0])
    lmul_indexes = jnp.array([1, 3, 4])
    expected_lmul = jnp.zeros_like(s.data).at[..., :3].set(s.data[..., lmul_indexes])

    lmul = rpj.ft_adjoint_left_mul(a, s)
    assert jnp.allclose(lmul.data, expected_lmul)

    # Right multiply is similar but with indexes 1, 3, 5
    rmul_indexes = jnp.array([1, 3, 5])
    expected_rmul = jnp.zeros_like(s.data).at[..., :3].set(s.data[..., rmul_indexes])

    rmul = rpj.ft_adjoint_right_mul(a, s)
    assert jnp.allclose(rmul.data, expected_rmul)


def test_adjoint_ft_mul_random_equivalent(rpj_dtype, rpj_batch, rpj_no_acceleration):
    rng = jax.random.key(12345)
    basis = rpj.TensorBasis(2, 2)

    x_data = jax.random.normal(rng, shape=(basis.size(),), dtype=rpj_dtype)
    x = rpj.DenseFreeTensor(x_data, basis)

    y_data = jax.random.normal(rng, shape=(basis.size(),), dtype=rpj_dtype)
    y = rpj.DenseFreeTensor(y_data, basis)

    shuffle = _random_shuffle_tensor(rng, basis, rpj_dtype, (basis.size(),))

    # Pair from dot products for checking adjoint equivalence <T*(x*), y> = <x*, T(y)>
    def pair_dot_data(lhs, rhs):
        return lhs.data.dot(rhs.data)

    lmul = pair_dot_data(rpj.ft_adjoint_left_mul(x, shuffle), y)
    expected_lmul = pair_dot_data(shuffle, rpj.ft_mul(x, y))
    assert jnp.allclose(lmul, expected_lmul)

    rmul = pair_dot_data(rpj.ft_adjoint_right_mul(x, shuffle), y)
    expected_rmul = pair_dot_data(shuffle, rpj.ft_mul(x, y))
    assert jnp.allclose(rmul, expected_rmul)
