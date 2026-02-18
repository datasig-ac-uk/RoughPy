import jax.numpy as jnp
import numpy as np
import roughpy_jax as rpj


def test_ft_mul(rpj_dtype, rpj_batch, rpj_no_acceleration):
    basis = rpj.TensorBasis(3, 3)

    a = rpj_batch.rng_nonzero_free_tensor(basis, rpj_dtype)
    b = rpj_batch.rng_nonzero_free_tensor(basis, rpj_dtype)

    result = rpj.ft_mul(a, b)

    z = rpj.FreeTensor(rpj_batch.zeros(basis.size(), rpj_dtype), basis)
    expected = rpj.ft_fma(z, a, b)

    assert jnp.allclose(result.data, expected.data)
