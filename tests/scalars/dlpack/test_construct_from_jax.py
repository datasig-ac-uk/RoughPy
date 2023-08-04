

import pytest

try:
    import jax
    from jax import numpy as jnp
except ImportError:
    jax = None
    jnp = None

import roughpy as rp
import numpy as np
from numpy.testing import assert_array_equal

@pytest.mark.skipif(jax is None, reason="JAX test require jax to be installed")
class TestJaxArrayCreation:

    @pytest.fixture(scope="class")
    def context(self):
        return rp.get_context(2, 2, rp.SPReal)

    @pytest.fixture(scope="class")
    def prng_key(self):
        return jax.random.PRNGKey(12345)

    @pytest.mark.xfail(condition=True, reason="No device support is currently available")
    def test_create_tensor_from_jax_array(self, prng_key, context):
        array = jax.random.uniform(prng_key, shape=(context.tensor_size(2),), dtype="float32", minval=-1.0, maxval=1.0)

        ts = rp.FreeTensor(array, ctx=context)

        assert_array_equal(np.array(ts), np.array(jax.device_get(array), copy=True))
