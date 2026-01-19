import jax
import jax.numpy as jnp
import numpy as np
import pytest
import roughpy_jax as rpj


# Running both f32 and f64 tests requires enabling JAX 64 bit mode
jax.config.update("jax_enable_x64", True)

# Set only CPU supported to prevent warnings in test output
jax.config.update("jax_platforms", "cpu")

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

        def st_f32(self, basis_width: int=2, basis_depth: int=2):
            basis = rpj.TensorBasis(basis_width, basis_depth)
            return rpj.ShuffleTensor(self.zeros_f32, basis)

        def st_f64(self, basis_width: int=2, basis_depth: int=2):
            basis = rpj.TensorBasis(basis_width, basis_depth)
            return rpj.ShuffleTensor(self.zeros_f64, basis)

        def st_i32(self, basis_width: int=2, basis_depth: int=2):
            basis = rpj.TensorBasis(basis_width, basis_depth)
            return rpj.ShuffleTensor(self.zeros_i32, basis)

    return BasisFixture()



# FIXME remove, use rpy_batch instead
@pytest.fixture(params=[(), (2,), (3, 2), (2, 2, 2)])
def batch_shape(request) -> tuple[int, ...]:
    return request.param


# Batching test fixture
@pytest.fixture(params=[(), (2,), (3, 2), (2, 2, 2)])
def rpy_batch(request):
    """
    Parameterised batch class fixture for various sizes with utility methods, example usage:

        def test_xs(rpy_batch):
            data = jnp.zeros(20)
            batched_data = rpy_batch.repeat(data)
            assert batched_data.shape[:-1] == rpy_batch.shape
    """
    class Batch:
        def __init__(self):
            self.rng = np.random.default_rng(1234)

        @property
        def shape(self):
            return request.param

        def tensor_batch_shape(self, basis):
            return (*self.shape, basis.size())

        def zeros(self, num, dtype):
            return self.repeat(jnp.zeros(num, dtype=dtype))

        def repeat(self, xs):
            return jnp.tile(xs, (*self.shape, 1))

        def rng_uniform(self, min, max, num, dtype):
            return self.rng.uniform(min, max, (*self.shape, num)).astype(dtype)

    return Batch()


# Data type test fixture
@pytest.fixture(params=[jnp.float32, jnp.float64])
def rpy_dtype(request):
    return request.param
