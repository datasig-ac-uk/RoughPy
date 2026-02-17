import jax.numpy as jnp
import roughpy_jax as rpj


def test_l2t_t2l_roundtrip(rpj_dtype, rpj_batch):
    lie_basis = rpj.LieBasis(4, 4)
    tensor_basis = rpj.TensorBasis(lie_basis.width, lie_basis.depth)

    x_data = rpj_batch.rng_uniform(-1, 1, lie_basis.size(), rpj_dtype)
    x = rpj.Lie(x_data, lie_basis)

    tensor_x = rpj.lie_to_tensor(x, tensor_basis)
    y = rpj.tensor_to_lie(tensor_x, lie_basis)

    assert y.data.dtype == x.data.dtype == rpj_dtype
    assert jnp.allclose(x.data, y.data, atol=1e-7)


def test_l2t_scaled_t2l_roundtrip(rpj_dtype, rpj_batch):
    lie_basis = rpj.LieBasis(4, 4)
    tensor_basis = rpj.TensorBasis(lie_basis.width, lie_basis.depth)

    x_data = rpj_batch.rng_uniform(-1, 1, lie_basis.size(), rpj_dtype)
    x = rpj.Lie(x_data, lie_basis)

    tensor_x = rpj.lie_to_tensor(x, tensor_basis, scale_factor=0.5)
    y = rpj.tensor_to_lie(tensor_x, lie_basis)

    assert y.data.dtype == x.data.dtype == rpj_dtype
    assert jnp.allclose(0.5 * x.data, y.data, atol=1e-7)


def test_l2t_t2l_scaled_roundtrip(rpj_dtype, rpj_batch):
    lie_basis = rpj.LieBasis(4, 4)
    tensor_basis = rpj.TensorBasis(lie_basis.width, lie_basis.depth)

    x_data = rpj_batch.rng_uniform(-1, 1, lie_basis.size(), rpj_dtype)
    x = rpj.Lie(x_data, lie_basis)

    tensor_x = rpj.lie_to_tensor(x, tensor_basis)
    y = rpj.tensor_to_lie(tensor_x, lie_basis, scale_factor=0.5)

    assert y.data.dtype == x.data.dtype == rpj_dtype
    assert jnp.allclose(0.5 * x.data, y.data, atol=1e-7)
