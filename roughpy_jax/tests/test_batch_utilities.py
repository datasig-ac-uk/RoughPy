import jax.numpy as jnp
import pytest

from roughpy_jax.dense_algebra import broadcast_to_batch_shape


def test_broadcast_to_batch_shape_scalar():
    # A scalar multiplier is reshaped so it can act uniformly across every
    # batch level of an algebra object, from the outermost batch axis inward.
    data = jnp.asarray(2.0, dtype=jnp.float32)

    result = broadcast_to_batch_shape(data, (3, 2))

    assert result.shape == (1, 1, 1)
    assert result.dtype == data.dtype


def test_broadcast_to_batch_shape_prefix_batch():
    # A partially batched scalar multiplier keeps its existing outer batch axis
    # and gains singleton axes so multiplication still broadcasts from the outside in.
    data = jnp.arange(3, dtype=jnp.float32)

    result = broadcast_to_batch_shape(data, (3, 2))

    assert result.shape == (3, 1, 1)
    expected = jnp.reshape(data, (3, 1, 1))
    assert jnp.array_equal(result, expected)


def test_broadcast_to_batch_shape_accepts_existing_core_singletons():
    # If a scalar multiplier is already shaped for the target batch structure,
    # the helper should leave it unchanged.
    data = jnp.ones((3, 2, 1), dtype=jnp.float32)

    result = broadcast_to_batch_shape(data, (3, 2))

    assert result.shape == data.shape
    assert jnp.array_equal(result, data)


def test_broadcast_to_batch_shape_rejects_incompatible_shape():
    data = jnp.ones((2, 2), dtype=jnp.float32)

    with pytest.raises(ValueError, match="incompatible"):
        broadcast_to_batch_shape(data, (3, 2))
