import jax.numpy as jnp
import roughpy_jax as rpj


# FIXME first draft copied from tests/compute/test_ft_fma.py
def test_dense_ft_fma():
    basis = rpj.TensorBasis(2, 2)
    a_data = jnp.zeros(basis.degree_begin.size, dtype=jnp.float32)
    b_data = jnp.array([2, 1, 3, 0.5, -1, 2, 0], dtype=jnp.float32)
    c_data = jnp.array([-1, 4, 0, 1, 1, 0, 2], dtype=jnp.float32)
    a = rpj.FreeTensor(a_data, basis)
    b = rpj.FreeTensor(b_data, basis)
    c = rpj.FreeTensor(c_data, basis)

    # FIXME remove dense_ prefix as in compute variant? e.g. compute.ft_fma(a, b, c)
    d = rpj.dense_ft_fma(a, b, c)
    print(d)

    # FIXME currently not getting same values as compute/test_ft_fma.py
    # zeroth order term: 2 . - 1
    # first order term: 2. (4 0) + -1 . (1 3)
    # and so on
    # expected = np.array([-2, 7, -3, 5.5, 3, 10, 4], dtype=np.float32)


def test_dense_ft_fma_compare_compute():
    # FIXME use roughpy_compute version and table of params
    pass