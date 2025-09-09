import jax.numpy as jnp
import roughpy_jax as rpj

def test_dense_ft_fma():
    basis = rpj.TensorBasis(width=2, depth=3)
    len = 100 # FIXME compute correctly

    a = rpj.FreeTensor(jnp.zeros(len), basis)
    b = rpj.FreeTensor(jnp.zeros(len), basis)
    c = rpj.FreeTensor(jnp.zeros(len), basis)

    d = rpj.dense_ft_fma(a, b, c)
    print(d)

    # FIXME compare with roughpy_compute