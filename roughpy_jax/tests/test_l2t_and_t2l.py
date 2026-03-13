import jax.numpy as jnp
import roughpy_jax as rpj
import pytest


def test_l2t_t2l_roundtrip(rpj_dtype, rpj_batch, rpj_no_acceleration):
    lie_basis = rpj.LieBasis(4, 4)
    tensor_basis = rpj.TensorBasis(lie_basis.width, lie_basis.depth)

    x_data = rpj_batch.rng_uniform(-1, 1, lie_basis.size(), rpj_dtype)
    x = rpj.Lie(x_data, lie_basis)

    tensor_x = rpj.lie_to_tensor(x, tensor_basis)
    y = rpj.tensor_to_lie(tensor_x, lie_basis)

    assert y.data.dtype == x.data.dtype == rpj_dtype
    assert jnp.allclose(x.data, y.data, atol=1e-7)


def test_l2t_scaled_t2l_roundtrip(rpj_dtype, rpj_batch, rpj_no_acceleration):
    lie_basis = rpj.LieBasis(4, 4)
    tensor_basis = rpj.TensorBasis(lie_basis.width, lie_basis.depth)

    x_data = rpj_batch.rng_uniform(-1, 1, lie_basis.size(), rpj_dtype)
    x = rpj.Lie(x_data, lie_basis)

    tensor_x = rpj.lie_to_tensor(x, tensor_basis, scale_factor=0.5)
    y = rpj.tensor_to_lie(tensor_x, lie_basis)

    assert y.data.dtype == x.data.dtype == rpj_dtype
    assert jnp.allclose(0.5 * x.data, y.data, atol=1e-7)


def test_l2t_t2l_scaled_roundtrip(rpj_dtype, rpj_batch, rpj_no_acceleration):
    lie_basis = rpj.LieBasis(4, 4)
    tensor_basis = rpj.TensorBasis(lie_basis.width, lie_basis.depth)

    x_data = rpj_batch.rng_uniform(-1, 1, lie_basis.size(), rpj_dtype)
    x = rpj.Lie(x_data, lie_basis)

    tensor_x = rpj.lie_to_tensor(x, tensor_basis)
    y = rpj.tensor_to_lie(tensor_x, lie_basis, scale_factor=0.5)

    assert y.data.dtype == x.data.dtype == rpj_dtype
    assert jnp.allclose(0.5 * x.data, y.data, atol=1e-7)
    
    
def test_l2t_t2l_l2t_roundtrip(rpj_dtype, rpj_batch, rpj_no_acceleration):
    lie_basis = rpj.LieBasis(4, 4)
    tensor_basis = rpj.TensorBasis(lie_basis.width, lie_basis.depth)

    x_data = rpj_batch.rng_uniform(-1, 1, lie_basis.size(), rpj_dtype)
    x = rpj.Lie(x_data, lie_basis)

    tensor_x = rpj.lie_to_tensor(x, tensor_basis)
    y = rpj.tensor_to_lie(tensor_x, lie_basis)
    tensor_y = rpj.lie_to_tensor(x, tensor_basis)

    assert jnp.allclose(tensor_x.data, tensor_y.data, atol=1e-7)


@pytest.mark.parametrize("scale_factor", [0.0, 0.5, 1.0, 2.0])
def test_l2t_scale_factor(rpj_batch, rpj_dtype, rpj_no_acceleration, scale_factor):
    # NOTE: the 0.0 scale factor should actually result in an identity tensor, 
    # but this test is still useful to check that the scaling is applied correctly. 
    
    """Test that a Lie is correctly scaled by the intersection length."""
    lie_basis = rpj.LieBasis(4, 4)
    tensor_basis = rpj.TensorBasis(lie_basis.width, lie_basis.depth)

    x_data = rpj_batch.rng_uniform(-1, 1, lie_basis.size(), rpj_dtype)
    x = rpj.Lie(x_data, lie_basis)
    
    l_prescaled = rpj.Lie(x.data * scale_factor, x.basis)
    t_prescaled = rpj.lie_to_tensor(l_prescaled, tensor_basis=tensor_basis)
    t_postscaled = rpj.lie_to_tensor(x, tensor_basis=tensor_basis, scale_factor=scale_factor)
    assert jnp.allclose(t_prescaled.data, t_postscaled.data, atol=1e-6)
    
    
@pytest.mark.parametrize("scale_factor", [0.0, 0.5, 1.0, 2.0])
def test_t2l_scale_factor(rpj_batch, rpj_dtype, rpj_no_acceleration, scale_factor):
    """Test that a Tensor is correctly scaled by the intersection length."""
    lie_basis = rpj.LieBasis(4, 4)
    tensor_basis = rpj.TensorBasis(lie_basis.width, lie_basis.depth)

    x_data = rpj_batch.rng_uniform(-1, 1, tensor_basis.size(), rpj_dtype)
    x = rpj.FreeTensor(x_data, tensor_basis)
    
    t = rpj.FreeTensor(x.data * scale_factor, tensor_basis)
    l_prescaled = rpj.tensor_to_lie(t, lie_basis=lie_basis)
    l_postscaled = rpj.tensor_to_lie(x, lie_basis=lie_basis, scale_factor=scale_factor)
    
    assert jnp.allclose(l_prescaled.data, l_postscaled.data, atol=1e-6)