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
    x:              (n,) dense vector data to multiply
    returns:        (n,) result of A @ x
    """
    # Generate a compact array of which non-zero cells to update
    cols = csc_cols_from_indptr(indptr)

    # Use a batched view of x so same logic applies to batched and flat arrays
    is_batched = x.ndim == 2
    x_batched = x if is_batched else x[:, None]

    # Calculate how much to add to sparse element (by indices). Reshaped data
    # allows broadcasting over batch dimensions
    updates = data[:, None] * x_batched[cols]

    # Accumulate the updates into batched output tensor data
    y = jnp.zeros((output_n_rows, x_batched.shape[1]), dtype=updates.dtype)
    y = y.at[indices, :].add(updates)

    # Unbatch the result if required
    return y if is_batched else y[:, 0]
