import jax.numpy as jnp
import pytest

from roughpy_jax.dense_algebra import broadcast_to_batch_shape


def test_broadcast_to_batch_shape_scalar():
    data = jnp.asarray(2.0, dtype=jnp.float32)

    result = broadcast_to_batch_shape(data, (3, 2))

    assert result.shape == (1, 1, 1)
    assert result.dtype == data.dtype


def test_broadcast_to_batch_shape_prefix_batch():
    data = jnp.arange(3, dtype=jnp.float32)

    result = broadcast_to_batch_shape(data, (3, 2))

    assert result.shape == (3, 1, 1)
    expected = jnp.reshape(data, (3, 1, 1))
    assert jnp.array_equal(result, expected)


def test_broadcast_to_batch_shape_rejects_incompatible_shape():
    data = jnp.ones((2, 2), dtype=jnp.float32)

    with pytest.raises(ValueError, match="incompatible"):
        broadcast_to_batch_shape(data, (3, 2))
