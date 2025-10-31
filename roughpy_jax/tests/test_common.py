import jax
import jax.numpy as jnp
import numpy as np
import pytest
import roughpy_jax as rpj


# Running both f32 and f64 tests requires enabling JAX 64 bit mode
jax.config.update("jax_enable_x64", True)


# Array types supported by roughpy_jax methods
array_dtypes = [jnp.float32, jnp.float64]


def jnp_to_np_float(jnp_dtype):
    """
    Unit test jnp to numpy float type conversion
    """
    if jnp_dtype == jnp.float32:
        return np.float32

    if jnp_dtype == jnp.float64:
        return np.float64

    raise ValueError("Expecting f32 or f64")


@pytest.fixture
def type_mismatch_fixture():
    """
    Unit test fixture for generating dummy arrays of given type. Reduces code
    duplication when writing type mismatch tests.
    """
    class BasisFixture:
        def __init__(self):
            self.zeros_f32 = jnp.zeros(100, dtype=jnp.float32)
            self.zeros_f64 = jnp.zeros(100, dtype=jnp.float64)
            self.zeros_i32 = jnp.zeros(100, dtype=jnp.int32)

        def ft_f32(self, basis_width: int=2, basis_depth: int=2):
            basis = rpj.TensorBasis(basis_width, basis_depth)
            return rpj.FreeTensor(self.zeros_f32, basis)

        def ft_f64(self, basis_width: int=2, basis_depth: int=2):
            basis = rpj.TensorBasis(basis_width, basis_depth)
            return rpj.FreeTensor(self.zeros_f64, basis)

        def ft_i32(self, basis_width: int=2, basis_depth: int=2):
            basis = rpj.TensorBasis(basis_width, basis_depth)
            return rpj.FreeTensor(self.zeros_i32, basis)

    return BasisFixture()





"""
def test_dense_ft_fma_dtype_mismatch():
    basis = rpj.TensorBasis(2, 2)

    # Mismatched array float types
    with pytest.raises(ValueError):
        rpj.ft_fma(
            rpj.FreeTensor(zeros_f32, basis),
            rpj.FreeTensor(zeros_f32, basis),
            rpj.FreeTensor(zeros_f64, basis)
        )

    # Unsupported array types
    with pytest.raises(ValueError):
        rpj.ft_fma(
            rpj.FreeTensor(zeros_f32, basis),
            rpj.FreeTensor(zeros_f32, basis),
            rpj.FreeTensor(zeros_i32, basis)
        )


@pytest.mark.parametrize("jnp_dtype", [jnp.float32, jnp.float64])
def test_dense_ft_fma(jnp_dtype):
    # Note, this test is a duplicate of tests/compute/test_ft_fma.py but for JAX
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


def test_dense_ft_fma_construction():
    rng = default_rng(12345)

    basis = compute.TensorBasis(2, 2)
    a = compute.FreeTensor(np.zeros(basis.size(), dtype=np.float32), basis)
    b = compute.FreeTensor(np.zeros(basis.size(), dtype=np.float32), basis)
    c = compute.FreeTensor(np.zeros(basis.size(), dtype=np.float32), basis)
    a.data = rng.uniform(-1, 1, size=basis.size())
    b.data = rng.uniform(-1, 1, size=basis.size())
    c.data = rng.uniform(-1, 1, size=basis.size())
    expected = a.data.copy()

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

    compute.ft_fma(a, b, c)
    assert_array_almost_equal(expected, a.data)
"""