import jax
import jax.numpy as jnp
import numpy as np
import pytest
import roughpy_jax as rpj


# Running both f32 and f64 tests requires enabling JAX 64 bit mode
jax.config.update("jax_enable_x64", True)


@pytest.fixture
def rpj_test_fixture_type_mismatch():
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

