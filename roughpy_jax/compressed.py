import jax.numpy as jnp

def expand_indptr(indptr):
    """
    Reconstruct indices from start array

    indptr:     row or column starts in indices and data
    returns:    (nnz,) expanded rows or columns where nnz = indptr[-1]
    """
    # Take the difference between adjacent element as number of repeats at each index,
    # e.g. in indptr [0 1 5 5 6] -> [0 1 1 1 1 3] second adjacent [1 5] generates four
    # 1s and next adjacent [5 5] generates zero 2s, so jumps straight to 3.
    lengths = indptr[1:] - indptr[:-1]
    return jnp.repeat(jnp.arange(lengths.shape[0], dtype=jnp.int32), lengths)


def csc_matvec(data, indices, indptr, n_rows, x):
    """
    Product of a compressed sparse column (CSC) matrix A and dense RHS x.

    data:    (nnz,)         stored values for A
    indices: (nnz,)         row indices for each stored value
    indptr:  (n_cols+1,)    column pointer array (indptr[-1] == nnz)
    n_rows:  int            number of rows in A
    x:       (*batch_dims, n_cols) dense vector(s) for the right-hand side
    returns: (*batch_dims, n_rows) result of A @ x
    """
    # Generate a compact array of which non-zero cells to update
    cols = expand_indptr(indptr)

    # Index x values given the columns, accommodating for leading batch dimensions
    # if present. For example a 1D batch of data x will be 2D (D1, N), and for a 2D
    # batch x will be 3D (D1, D2, N).
    x_by_col = x[..., cols]

    # Updates are all the sparse additions at every location in all tensors
    updates = data * x_by_col

    # Accumulate the updates into batched output tensor data, taking x's leading
    # dimensions as the batch shape; the tensors themselves are always 1D trailing.
    batched_shape = x_by_col.shape[:-1]
    y = jnp.zeros((*batched_shape, n_rows), dtype=updates.dtype)

    # Scatter-add updates using indices, i.e y[..., indices[i]] += updates[..., i]
    y = y.at[..., indices].add(updates)
    return y


def csr_matvec(data, indices, indptr, n_cols, x):
    """
    Product of a compressed sparse row (CSR) matrix A and dense RHS x.

    data:    (nnz,)         stored values for A
    indices: (nnz,)         column indices for each stored value
    indptr:  (n_rows+1,)    row pointer array (indptr[-1] == nnz)
    n_cols:  int            number of columns in A
    x:       (*batch_dims, n_cols) dense vector(s) for the right-hand side
    returns: (*batch_dims, n_rows) result of A @ x
    """
    # Compact array of cells to update
    rows = expand_indptr(indptr)

    # Gather x values by column indices, preserving batch dims
    x_by_col = x[..., indices]      # shape: (*batch_dims, nnz)

    # Updates are all the sparse additions at every location in all tensors
    updates = data * x_by_col       # broadcasting over batch dims

    # Output array to receive stattered updated (*batch_dims, n_rows)
    batched_shape = x_by_col.shape[:-1]
    y = jnp.zeros((*batched_shape, n_cols), dtype=updates.dtype)

    # Scatter-add to accumulate into rows: y[..., rows[i]] += updates[..., i]
    y = y.at[..., rows].add(updates)
    return y
