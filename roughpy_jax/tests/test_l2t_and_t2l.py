import jax.numpy as jnp
import numpy as np
import pytest
import roughpy_jax as rpj

from rpy_test_common import array_dtypes


# FIXME replace with new batch methods in conftest.py
batch_dims = [(), (10,), (10, 5)]

@pytest.mark.parametrize("jnp_dtype", array_dtypes)
@pytest.mark.parametrize("batch_dims", batch_dims)
def test_l2t_t2l_roundtrip(jnp_dtype, batch_dims):
    rng = np.random.default_rng(1234)

    lie_basis = rpj.LieBasis(4, 4)
    tensor_basis = rpj.TensorBasis(lie_basis.width, lie_basis.depth)

    x_data = rng.uniform(-1.0, 1.0, (*batch_dims, lie_basis.size())).astype(jnp_dtype)
    x = rpj.Lie(x_data, lie_basis)

    tensor_x = rpj.lie_to_tensor(x, tensor_basis)
    y = rpj.tensor_to_lie(tensor_x, lie_basis)

    assert y.data.dtype == x.data.dtype == jnp_dtype
    assert jnp.allclose(x.data, y.data, atol=1e-7)


@pytest.mark.parametrize("jnp_dtype", array_dtypes)
@pytest.mark.parametrize("batch_dims", batch_dims)
def test_l2t_scaled_t2l_roundtrip(jnp_dtype, batch_dims):
    rng = np.random.default_rng(1234)

    lie_basis = rpj.LieBasis(4, 4)
    tensor_basis = rpj.TensorBasis(lie_basis.width, lie_basis.depth)

    x_data = rng.uniform(-1.0, 1.0, (*batch_dims, lie_basis.size())).astype(jnp_dtype)
    x = rpj.Lie(x_data, lie_basis)

    tensor_x = rpj.lie_to_tensor(x, tensor_basis, scale_factor=0.5)
    y = rpj.tensor_to_lie(tensor_x, lie_basis)

    assert y.data.dtype == x.data.dtype == jnp_dtype
    assert jnp.allclose(0.5 * x.data, y.data, atol=1e-7)


@pytest.mark.parametrize("jnp_dtype", array_dtypes)
@pytest.mark.parametrize("batch_dims", batch_dims)
def test_l2t_t2l_scaled_roundtrip(jnp_dtype, batch_dims):
    rng = np.random.default_rng(1234)

    lie_basis = rpj.LieBasis(4, 4)
    tensor_basis = rpj.TensorBasis(lie_basis.width, lie_basis.depth)

    x_data = rng.uniform(-1.0, 1.0, (*batch_dims, lie_basis.size())).astype(jnp_dtype)
    x = rpj.Lie(x_data, lie_basis)

    tensor_x = rpj.lie_to_tensor(x, tensor_basis)
    y = rpj.tensor_to_lie(tensor_x, lie_basis, scale_factor=0.5)

    assert y.data.dtype == x.data.dtype == jnp_dtype
    assert jnp.allclose(0.5 * x.data, y.data, atol=1e-7)
