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

    # Coerce the data into a shape that can be broadcasted to given the shape of x, which may
    # be unbatched (N,) or batched any number of dimensions, e.g. (N, D1), (N, D1, D2) etc.
    # The reshaped data is a view of the original with trailing 1s to the right allowing
    # broadcast in multiplication, e.g. for a 3x4 2D batch of tensors of length 10,
    # data_reshaped would be (10, 1, 1) for broadcast from x_by_col (10, 3, 4)
    x_by_col = x[cols]
    missing_dims = x_by_col.ndim - data.ndim
    data_shape = (data.shape[0],) + (1,) * missing_dims
    data_reshaped = jnp.reshape(data, data_shape)

    # The updates are all the sparse additions at every location in all tensors
    updates = data_reshaped * x_by_col

    # Accumulate the updates into batched output tensor data
    batched_shape = x_by_col.shape[1:]
    y = jnp.zeros((output_n_rows, *batched_shape), dtype=updates.dtype)
    y = y.at[indices, ...].add(updates)
    return y
