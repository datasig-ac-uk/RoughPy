import jax
import jax.numpy as jnp
import roughpy_jax as rpj
import time


# Arbitrary combination of ft_fma for comparison and timing
def _combined_fma(a, b, c):
    d = rpj.ft_fma(a, b, c)
    e = rpj.ft_fma(b, c, d)
    f = rpj.ft_fma(c, d, e)
    g = rpj.ft_fma(d, e, f)
    h = rpj.ft_fma(e, f, g)
    return h


def test_jit_compilation(rpj_dtype, rpj_batch, rpj_no_acceleration):
    # Minimal test JIT-ing CPU and fallback code to detect tracer issues
    _combined_fma_jit = jax.jit(_combined_fma)


def test_jit_equivalence(rpj_dtype, rpj_batch):
    basis = rpj.TensorBasis(3, 3)
    data = rpj_batch.rng_uniform(-1, 1, basis.size(), rpj_dtype)
    a = rpj.FreeTensor(data, basis)
    b = rpj.FreeTensor(data, basis)
    c = rpj.FreeTensor(data, basis)

    # Confirm that results are same for JIT and non-JIT runs
    result_no_jit = _combined_fma(a, b, c)
    combined_fma_jit = jax.jit(_combined_fma)
    result_jit = combined_fma_jit(a, b, c)
    assert jnp.allclose(result_no_jit.data, result_jit.data)

    # Confirm that JIT is running faster
    time_non_jit_start = time.time()
    for _ in range(100):
        _combined_fma(a, b, c)
    time_non_jit = time.time() - time_non_jit_start

    time_jit_start = time.time()
    for _ in range(100):
        combined_fma_jit(a, b, c)
    time_jit = time.time() - time_jit_start

    assert time_jit < time_non_jit
