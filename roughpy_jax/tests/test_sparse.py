import jax.numpy as jnp
import pytest

from roughpy_jax.compressed import expand_indptr, csc_matvec, csr_matvec


def _indptr_from_outer_indices(cols):
    """Regenerate indptr from column or row test data"""
    # Count the number of times each column/row occurs and the cumulatively sum.
    # e.g. cols = [1, 1, 2, 2, 2, 4], counts = [0, 2, 3, 0, 1], then cumulative
    # sum is [0, 2, 5, 5, 6]. Prefix with [0] for first CSC/CSR data index.
    counts = jnp.bincount(cols, length=cols.max() + 1)
    return jnp.concatenate([jnp.array([0]), jnp.cumsum(counts)])


class SampleMatrixFixture:
    def __init__(self):
        # COO Sparse representation of
        #   | 1   0   0 | 
        #   | 0   3   0 | 
        #   | 0   2  -1 |
        self.rows = jnp.array([0, 1, 2, 2], dtype=jnp.int32)
        self.cols = jnp.array([0, 1, 1, 2], dtype=jnp.int32)
        self.data = jnp.array([1.0, 3.0, 2.0, -1.0], dtype=jnp.float32)

        # Dense for reference
        self.dense = jnp.zeros((3, 3)).at[self.rows, self.cols].set(self.data)

        # Matrix default dims
        self.n_rows = int(self.rows.max()) + 1
        self.n_cols = int(self.cols.max()) + 1

        # Columns and row pointers for CSR
        self.csr_indices = self.cols
        self.csr_indptr = _indptr_from_outer_indices(self.rows)

        # Rows and column pointers for CSC
        self.csc_indices = self.rows
        self.csc_indptr = _indptr_from_outer_indices(self.cols)

    def matvec_args(self, fmt):
        is_csr = (fmt == "csr")
        return [
            self.data,
            self.csr_indices if is_csr else self.csc_indices,
            self.csr_indptr if is_csr else self.csc_indptr,
            self.n_cols if is_csr else self.n_rows
        ]


@pytest.fixture
def sample_matrix():
    return SampleMatrixFixture()


def test_sample_matrix_fixture(sample_matrix):
    assert jnp.array_equal(sample_matrix.csc_indptr, jnp.array([0, 1, 3, 4]))
    assert jnp.array_equal(sample_matrix.csr_indptr, jnp.array([0, 1, 2, 4]))


def test_sparse_expand_indptr():
    indptr = jnp.array([0, 1, 5, 5, 6])
    cols = expand_indptr(indptr)

    expected_cols = jnp.array([0, 1, 1, 1, 1, 3])
    assert jnp.array_equal(cols, expected_cols)


@pytest.mark.parametrize("fmt", ["csr", "csc"])
def test_sparse_matvec(sample_matrix, fmt):
    # Test sparse matrix mult y = A @ x
    matvec_args = sample_matrix.matvec_args(fmt)
    matvec_fn = csr_matvec if fmt == "csr" else csc_matvec

    # x = |1 2 3| gives y = |1 6 1|
    x = jnp.array([1, 2, 3])
    y = matvec_fn(*matvec_args, x)
    expected_y = jnp.array([1, 6, 1])
    assert jnp.array_equal(y, expected_y)


@pytest.mark.parametrize("fmt", ["csr", "csc"])
def test_sparse_csr_matvec_roundtrip(sample_matrix, fmt):
    matvec_args = sample_matrix.matvec_args(fmt)
    matvec_fn = csr_matvec if fmt == "csr" else csc_matvec

    x = jnp.array([10., 20., 30.])
    y = matvec_fn(*matvec_args, x)
    assert jnp.allclose(y, sample_matrix.dense @ x)

    xb = jnp.stack([x, x * 0.1])
    yb = matvec_fn(*matvec_args, xb)
    assert jnp.allclose(yb, xb @ sample_matrix.dense.T)
