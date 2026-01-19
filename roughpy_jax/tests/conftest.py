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


# Batching test fixture
@pytest.fixture(params=[(), (2,), (3, 2), (2, 2, 2)])
def rpj_batch(request):
    """
    Parameterised batch class fixture for various sizes with utility methods, example usage:

        def test_xs(rpj_batch):
            data = jnp.zeros(20)
            batched_data = rpj_batch.repeat(data)
            assert batched_data.shape[:-1] == rpj_batch.shape
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

        def rng_nonzero_free_tensor(self, basis, dtype):
            """
            Several tests need to operate on non-zero values otherwise exp/log will
            end up always being zero and tests are not doing anything useful. Only the
            vector part (directly after scalar part, hence [1:width+1]) needs to be set
            to ensure the overall value does not collapse to zero.
            """
            # Built using np not jnp for easy mutability
            data = np.zeros(self.tensor_batch_shape(basis), dtype)
            data[...,1:basis.width + 1] = self.rng.normal(size=(*self.shape, basis.width))
            return rpj.FreeTensor(data, basis)

        def identity_zero_data(self, basis, dtype):
            # Built using np not jnp for easy mutability
            data = np.zeros(self.tensor_batch_shape(basis), dtype)
            data[...,0] = 1.0
            return data

    return Batch()


# Data type test fixture
@pytest.fixture(params=[jnp.float32, jnp.float64])
def rpj_dtype(request):
    return request.param
