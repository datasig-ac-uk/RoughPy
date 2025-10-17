import jax.numpy as jnp
import pytest
import roughpy_jax as rpj


def test_dense_ft_fma_basis_mismatch():
    zeros = jnp.zeros(100, dtype=jnp.float32)

    # Mismatch first and second widths
    with pytest.raises(ValueError):
        rpj.ft_fma(
            rpj.FreeTensor(zeros, rpj.TensorBasis(2, 2)),
            rpj.FreeTensor(zeros, rpj.TensorBasis(3, 2)),
            rpj.FreeTensor(zeros, rpj.TensorBasis(2, 2))
        )

    # Mismatch first and third widths
    with pytest.raises(ValueError):
        rpj.ft_fma(
            rpj.FreeTensor(zeros, rpj.TensorBasis(2, 2)),
            rpj.FreeTensor(zeros, rpj.TensorBasis(2, 2)),
            rpj.FreeTensor(zeros, rpj.TensorBasis(3, 2))
        )


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
    d = rpj.ft_fma(a, b, c)

    expected = jnp.array([-2, 7, -3, 5.5, 3, 10, 4], dtype=jnp.float32)
    assert jnp.allclose(d.data, expected)


def test_dense_ft_fma_compare_compute():
    # FIXME use roughpy_compute version and table of params
    pass