import jax.numpy as jnp
import pytest
import roughpy_jax as rpj


def test_dense_st_ft_pairing(rpj_batch):
    basis = rpj.TensorBasis(2, 2)
    dtype = jnp.dtype("float32")

    functional = rpj_batch.rng_shuffle_tensor(basis, dtype)
    argument = rpj_batch.rng_nonzero_free_tensor(basis, dtype)

    result = rpj.tensor_pairing(functional, argument)

    assert result.shape == rpj_batch.shape
    assert result.dtype == dtype

    expected = jnp.einsum("...i,...i -> ...", functional.data, argument.data)
    assert jnp.allclose(result, expected)
