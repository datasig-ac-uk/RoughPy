import jax.numpy as jnp
import numpy as np
import pytest
import roughpy_jax as rpj

from rpy_test_common import array_dtypes


@pytest.mark.skipif(True, reason="FIXME disabled awaiting lie tensor internals")
@pytest.mark.parametrize("jnp_dtype", array_dtypes)
def test_l2t_t2l_roundtrip(jnp_dtype):
    lie_basis = rpj.LieBasis(4, 4)
    tensor_basis = rpj.TensorBasis(lie_basis.width, lie_basis.depth)

    rng = np.random.default_rng()

    x_data = rng.uniform(-1.0, 1.0, (lie_basis.size(),))

    x = rpj.Lie(x_data, lie_basis)

    tensor_x = rpj.lie_to_tensor(x, tensor_basis)

    y = rpj.tensor_to_lie(tensor_x, lie_basis)

    assert jnp.allclose(x.data, y.data, atol=1e-7)


@pytest.mark.skipif(True, reason="FIXME disabled awaiting lie tensor internals")
@pytest.mark.parametrize("jnp_dtype", array_dtypes)
def test_l2t_scaled_t2l_roundtrip(jnp_dtype):
    lie_basis = rpj.LieBasis(4, 4)
    tensor_basis = rpj.TensorBasis(lie_basis.width, lie_basis.depth)

    rng = np.random.default_rng()

    x_data = rng.uniform(-1.0, 1.0, (lie_basis.size(),))

    x = rpj.Lie(x_data, lie_basis)

    tensor_x = rpj.lie_to_tensor(x, tensor_basis, scale_factor=0.5)

    y = rpj.tensor_to_lie(tensor_x, lie_basis)

    assert jnp.allclose(0.5 * x.data, y.data, atol=1e-7)


@pytest.mark.skipif(True, reason="FIXME disabled awaiting lie tensor internals")
@pytest.mark.parametrize("jnp_dtype", array_dtypes)
def test_l2t_t2l_scaled_roundtrip(jnp_dtype):
    lie_basis = rpj.LieBasis(4, 4)
    tensor_basis = rpj.TensorBasis(lie_basis.width, lie_basis.depth)

    rng = np.random.default_rng()

    x_data = rng.uniform(-1.0, 1.0, (lie_basis.size(),))

    x = rpj.Lie(x_data, lie_basis)

    tensor_x = rpj.lie_to_tensor(x, tensor_basis)

    y = rpj.tensor_to_lie(tensor_x, lie_basis, scale_factor=0.5)

    assert jnp.allclose(0.5 * x.data, y.data, atol=1e-7)
