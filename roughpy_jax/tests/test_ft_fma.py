import jax.numpy as jnp
import roughpy_jax as rpj


def test_dense_ft_fma():
    # Note, this test is a duplicate of tests/compute/test_ft_fma.py but for JAX
    basis = rpj.TensorBasis(2, 2)
    a_data = jnp.zeros(basis.size(), dtype=jnp.float32)
    b_data = jnp.array([2, 1, 3, 0.5, -1, 2, 0], dtype=jnp.float32)
    c_data = jnp.array([-1, 4, 0, 1, 1, 0, 2], dtype=jnp.float32)
    a = rpj.FreeTensor(a_data, basis)
    b = rpj.FreeTensor(b_data, basis)
    c = rpj.FreeTensor(c_data, basis)

    # FIXME Should this be named dense_ft_fma or just ft_fma as in roughpy_compute?
    d = rpj.dense_ft_fma(a, b, c)

    expected = jnp.array([-2, 7, -3, 5.5, 3, 10, 4], dtype=jnp.float32)
    jnp.allclose(d, expected)


def test_dense_ft_fma_compare_compute():
    # FIXME use roughpy_compute version and table of params
    pass