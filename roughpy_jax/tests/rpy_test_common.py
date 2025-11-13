import jax.numpy as jnp
import numpy as np


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
