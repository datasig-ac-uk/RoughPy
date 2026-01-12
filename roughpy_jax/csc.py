import jax.numpy as jnp

def csc_cols_from_indptr(indptr):
    """
    Reconstruct columns indices from column start array

    indptr:     column starts in indices and data
    returns:    (n,) colums where n = indptr[-1]
    """
    # Take the difference between adjacent element as number of repeats at each index, e.g.
    # indptr [0 1 5 5 6] -> cols [0 1 1 1 1 3], e.g. second adjacent 1-5 generates four 1s,
    # and next adjacent 5-5 generates zero 2s, so jumps straight to 3.
    lengths = indptr[1:] - indptr[:-1]
    return jnp.repeat(jnp.arange(lengths.shape[0], dtype=jnp.int32), lengths)


def csc_matvec(data, indices, indptr, output_n_rows, x):
    """
    Product of compressed sparse column matrix and dense matrix

    data:           sparse matrix A nonzero values
    indices:        sparse matrix A row indices
    indptr:         sparse matrix A column
    out_rows:       number of rows in result (n)
    x:              (*batch_dims, n,) dense vector data to multiply
    returns:        (*batch_dims, n,) result of A @ x
    """
    # Generate a compact array of which non-zero cells to update
    cols = csc_cols_from_indptr(indptr)

    # Index x values given the columns, accommodating for leading batch dimensions
    # if present. For example a 1D batch of data x will be 2D (D1, N), and for a 2D
    # batch x will be 3D (D1, D2, N).
    x_by_col = x[..., cols]

    # Updates are all the sparse additions at every location in all tensors
    updates = data * x_by_col

    # Accumulate the updates into batched output tensor data, taking x's leading
    # dimensions as the batch shape; the tensors themselves are always 1D trailing.
    batched_shape = x_by_col.shape[:-1]
    y = jnp.zeros((*batched_shape, output_n_rows), dtype=updates.dtype)

    # Scatter-add updates using indices, i.e y[..., indices[i]] += updates[..., i]
    y = y.at[..., indices].add(updates)
    return y
