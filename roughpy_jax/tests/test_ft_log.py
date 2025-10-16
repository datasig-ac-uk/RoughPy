import jax.numpy as jnp
import roughpy_jax as rpj


def test_dense_ft_log():
    basis = rpj.TensorBasis(2, 2)
    x_data = jnp.array([2, 1, 3, 0.5, -1, 2, 0], dtype=jnp.float32)
    x = rpj.FreeTensor(x_data, basis)

    a = rpj.dense_ft_log(x)

    # FIXME assert result, try other basis
    print(a)
