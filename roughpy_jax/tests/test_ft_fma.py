import jax.numpy as jnp
import numpy as np
import pytest
import roughpy_jax as rpj

from test_common import type_mismatch_fixture, array_dtypes, jnp_to_np_float

# FIXME add JIT tests

def test_dense_ft_fma_array_mismatch(type_mismatch_fixture):
    f = type_mismatch_fixture

    # Mismatch first and second widths
    with pytest.raises(ValueError):
        rpj.ft_fma(f.ft_f32(2, 2), f.ft_f32(3, 2), f.ft_f32(2, 2))

    # Mismatch first and third widths
    with pytest.raises(ValueError):
        rpj.ft_fma(f.ft_f32(2, 2), f.ft_f32(2, 2), f.ft_f32(3, 2))

    # Mismatched array float types
    with pytest.raises(ValueError):
        rpj.ft_fma(f.ft_f32(), f.ft_f64(), f.ft_f32())

    # Unsupported array types
    with pytest.raises(ValueError):
        rpj.ft_fma(f.ft_f32(), f.ft_f32(), f.ft_i32())


@pytest.mark.parametrize("jnp_dtype", array_dtypes)
def test_dense_ft_fma(jnp_dtype):
    basis = rpj.TensorBasis(2, 2)
    a_data = jnp.zeros(basis.size(), dtype=jnp_dtype)
    b_data = jnp.array([2, 1, 3, 0.5, -1, 2, 0], dtype=jnp_dtype)
    c_data = jnp.array([-1, 4, 0, 1, 1, 0, 2], dtype=jnp_dtype)
    a = rpj.FreeTensor(a_data, basis)
    b = rpj.FreeTensor(b_data, basis)
    c = rpj.FreeTensor(c_data, basis)

    d = rpj.ft_fma(a, b, c)
    expected = jnp.array([-2, 7, -3, 5.5, 3, 10, 4], dtype=jnp_dtype)
    assert jnp.allclose(d.data, expected)


@pytest.mark.parametrize("jnp_dtype", array_dtypes)
def test_dense_ft_fma_construction(jnp_dtype):
    rng = np.random.default_rng(12345)
    basis = rpj.TensorBasis(2, 2)

    def _create_rng_uniform_ft():
        data = np.zeros(basis.size(), dtype=jnp_to_np_float(jnp_dtype))
        data = rng.uniform(-1, 1, size=basis.size())
        return rpj.FreeTensor(data, basis)

    a = _create_rng_uniform_ft()
    b = _create_rng_uniform_ft()
    c = _create_rng_uniform_ft()
    expected = np.array(a.data)

    # scalar term
    expected[0] += b.data[0] * c.data[0]

    # first order term
    expected[1:3] += b.data[0] * c.data[1:3] + c.data[0] * b.data[1:3]

    # second order term
    expected[3:] += (
        b.data[0] * c.data[3:]
        + c.data[0] * b.data[3:]
        + np.outer(b.data[1:3], c.data[1:3]).flatten()
    )

    d = rpj.ft_fma(a, b, c)
    assert jnp.allclose(d.data, expected)
