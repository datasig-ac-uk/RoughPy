import jax.numpy as jnp

from roughpy_jax.csc import csc_cols_from_indptr, csc_matvec


def _indptr_from_csc_cols(cols):
    """Regenerate indptr from cols test data"""
    # Count the number of times each column occurs and the cumulatively sum.
    # e.g. cols = [1, 1, 2, 2, 2, 4], counts = [0, 2, 3, 0, 1], then cumulative
    # sum is [0, 2, 5, 5, 6]. Prefix with [0] for first CSC data index.
    counts = jnp.bincount(cols, length=cols.max() + 1)
    return jnp.concatenate([jnp.array([0]), jnp.cumsum(counts)])


def test_csc_cols_from_indptr():
    indptr = jnp.array([0, 1, 5, 5, 6])
    cols = csc_cols_from_indptr(indptr)

    expected_cols = jnp.array([0, 1, 1, 1, 1, 3])
    assert jnp.array_equal(cols, expected_cols)


def test_csc_matvec():
    # sparse representation of 
    #   | 1   0   0 | 
    #   | 0   3   0 | 
    #   | 0   2  -1 |
    rows = jnp.array([0, 1, 2, 2])
    cols = jnp.array([0, 1, 1, 2])
    data = jnp.array([1, 3, 2, -1])

    # Get indptr from cols and sanity check it matches original
    indptr = _indptr_from_csc_cols(cols)
    expected_cols = csc_cols_from_indptr(indptr)
    assert jnp.array_equal(cols, expected_cols)

    # Test matrix mult by |1 2 3| = |1 6 1|
    x = jnp.array([1, 2, 3])
    y = csc_matvec(data, rows, indptr, 3, x)
    expected_y = jnp.array([1, 6, 1])
    assert jnp.array_equal(y, expected_y)
