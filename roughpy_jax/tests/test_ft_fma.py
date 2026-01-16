import jax
import jax.numpy as jnp
import numpy as np
import pytest
import roughpy_jax as rpj
import time


def test_dense_ft_fma_array_mismatch(rpj_test_fixture_type_mismatch):
    f = rpj_test_fixture_type_mismatch

    # Mismatch first and second widths
    with pytest.raises(ValueError):
        rpj.ft_fma(f.ft_f32(2, 2), f.ft_f32(3, 2), f.ft_f32(2, 2))

    # Mismatch first and third widths
    with pytest.raises(ValueError):
        rpj.ft_fma(f.ft_f32(2, 2), f.ft_f32(2, 2), f.ft_f32(3, 2))

    # Mismatched array float types
    with pytest.raises(ValueError):
        rpj.ft_fma(f.ft_f32(), f.ft_f64(), f.ft_f32())

    # Unsupported array types
    with pytest.raises(ValueError):
        rpj.ft_fma(f.ft_i32(), f.ft_i32(), f.ft_i32())


def test_dense_ft_mul_array_mismatch(rpj_test_fixture_type_mismatch):
    f = rpj_test_fixture_type_mismatch

    # Mismatch first and second widths
    with pytest.raises(ValueError):
        rpj.ft_mul(f.ft_f32(2, 2), f.ft_f32(3, 2))

    # Mismatched array float types
    with pytest.raises(ValueError):
        rpj.ft_mul(f.ft_f32(), f.ft_f64())

    # Unsupported array types
    with pytest.raises(ValueError):
        rpj.ft_mul(f.ft_i32(), f.ft_i32())


def test_dense_ft_fma(rpy_dtype, rpy_batch):
    basis = rpj.TensorBasis(2, 2)
    a_data = rpy_batch.zeros(basis.size(), rpy_dtype)
    b_data = rpy_batch.repeat(jnp.array([2, 1, 3, 0.5, -1, 2, 0], dtype=rpy_dtype))
    c_data = rpy_batch.repeat(jnp.array([-1, 4, 0, 1, 1, 0, 2], dtype=rpy_dtype))

    a = rpj.FreeTensor(a_data, basis)
    b = rpj.FreeTensor(b_data, basis)
    c = rpj.FreeTensor(c_data, basis)

    d = rpj.ft_fma(a, b, c)

    expected_data = rpy_batch.repeat(jnp.array([-2, 7, -3, 5.5, 3, 10, 4], dtype=rpy_dtype))
    assert jnp.allclose(d.data, expected_data)


def test_dense_ft_fma_construction(rpy_dtype, rpy_batch):
    basis = rpj.TensorBasis(2, 2)

    def _create_rng_uniform_ft():
        data = rpy_batch.rng_uniform(-1, 1, basis.size(), rpy_dtype)
        return rpj.FreeTensor(data, basis)

    batched_a = _create_rng_uniform_ft()
    batched_b = _create_rng_uniform_ft()
    batched_c = _create_rng_uniform_ft()

    batched_d = rpj.ft_fma(batched_a, batched_b, batched_c)

    # FIXME for review, iterating over all combinations to preserve old expected computation 
    for idx in np.ndindex(rpy_batch.shape):
        a = batched_a.data[idx]
        b = batched_b.data[idx]
        c = batched_c.data[idx]
        d = batched_d.data[idx]

        expected = np.array(a)

        # scalar term
        expected[0] += b[0] * c[0]

        # first order term
        expected[1:3] += b[0] * c[1:3] + c[0] * b[1:3]

        # second order term
        expected[3:] += (
            b[0] * c[3:]
            + c[0] * b[3:]
            + np.outer(b[1:3], c[1:3]).flatten()
        )

        assert jnp.allclose(d, expected)


def test_ft_fma_jit(rpy_dtype, rpy_batch):
    # Arbitrary combination of ft_fma for comparison and timing
    def combined_fma(a, b, c):
        d = rpj.ft_fma(a, b, c)
        e = rpj.ft_fma(b, c, d)
        f = rpj.ft_fma(c, d, e)
        g = rpj.ft_fma(d, e, f)
        h = rpj.ft_fma(e, f, g)
        return h

    basis = rpj.TensorBasis(3, 3)
    data = rpy_batch.rng_uniform(-1, 1, basis.size(), rpy_dtype)
    a = rpj.FreeTensor(data, basis)
    b = rpj.FreeTensor(data, basis)
    c = rpj.FreeTensor(data, basis)

    # Confirm that results are same for JIT and non-JIT runs
    result_no_jit = combined_fma(a, b, c)
    combined_fma_jit = jax.jit(combined_fma)
    result_jit = combined_fma_jit(a, b, c)
    assert jnp.allclose(result_no_jit.data, result_jit.data)

    # Confirm that JIT is running faster
    time_non_jit_start = time.time()
    for _ in range(100):
        combined_fma(a, b, c)
    time_non_jit = time.time() - time_non_jit_start
    print(f"ft_fma non-JIT time: {time_non_jit}")

    time_jit_start = time.time()
    for _ in range(100):
        combined_fma_jit(a, b, c)
    time_jit = time.time() - time_jit_start
    print(f"ft_fma JIT time: {time_jit}")

    assert time_jit < time_non_jit
