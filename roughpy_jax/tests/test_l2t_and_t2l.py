import jax.numpy as jnp
import pytest
import roughpy_jax as rpj

from derivative_testing import (
    DerivativeTrialsHelper,
    assert_is_adjoint_derivative,
    assert_is_derivative,
    assert_is_linear,
)


# Test fixture for batch of w=4, d=4 lie tensor
@pytest.fixture(params=[jnp.float32, jnp.float64])
def lt2_trials(request):
    yield DerivativeTrialsHelper(request.param, width=4, depth=4)


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


def test_l2t_linear(lt2_trials):
    x = lt2_trials.uniform_lie()
    y = lt2_trials.uniform_lie()

    vals = lt2_trials.uniform_data((2,))
    alpha = float(vals[0])
    beta = float(vals[1])

    assert_is_linear(rpj.lie_to_tensor, x, y, alpha, beta)


def test_l2t_derivative(lt2_trials):
    x = lt2_trials.uniform_lie()
    tangent = lt2_trials.uniform_lie() * lt2_trials.cond_dtype(1e-3, 1e0)

    assert_is_derivative(
        rpj.lie_to_tensor,
        rpj.lie_to_tensor_derivative,
        x,
        tangent,
        abs_tol=lt2_trials.cond_dtype(5e-2, 1e-6),
    )


def test_l2t_adjoint_derivative(lt2_trials):
    x = lt2_trials.uniform_lie()
    tangent = lt2_trials.uniform_lie() * lt2_trials.cond_dtype(1e-3, 1e0)
    cotangent = lt2_trials.uniform_shuffle_tensor()

    def pairing(lhs, rhs):
        return jnp.sum(lhs.data * rhs.data)

    assert_is_adjoint_derivative(
        rpj.lie_to_tensor,
        rpj.lie_to_tensor_adjoint_derivative,
        x,
        tangent,
        cotangent,
        domain_pairing=pairing,
        codomain_pairing=pairing,
        abs_tol=lt2_trials.cond_dtype(5e-2, 1e-6),
    )


def test_t2l_linear(lt2_trials):
    lie_basis = lt2_trials.lie_basis
    x = lt2_trials.uniform_free_tensor()
    y = lt2_trials.uniform_free_tensor()

    vals = lt2_trials.uniform_data((2,))
    alpha = float(vals[0])
    beta = float(vals[1])

    assert_is_linear(rpj.tensor_to_lie, x, y, alpha, beta)


def test_t2l_derivative(lt2_trials):
    lie_basis = lt2_trials.lie_basis
    x = lt2_trials.uniform_free_tensor()
    tangent = lt2_trials.uniform_free_tensor() * lt2_trials.cond_dtype(1e-3, 1e0)
    
    assert_is_derivative(
        rpj.tensor_to_lie,
        rpj.tensor_to_lie_derivative,
        x,
        tangent,
        abs_tol=lt2_trials.cond_dtype(5e-2, 1e-6),
    )


def test_t2l_adjoint_derivative(lt2_trials):
    lie_basis = lt2_trials.lie_basis
    x = lt2_trials.uniform_free_tensor()
    tangent = lt2_trials.uniform_free_tensor() * lt2_trials.cond_dtype(1e-3, 1e0)
    cotangent = lt2_trials.uniform_lie()

    def pairing(lhs, rhs):
        return jnp.sum(lhs.data * rhs.data)
    
    assert_is_adjoint_derivative(
        rpj.tensor_to_lie,
        rpj.tensor_to_lie_adjoint_derivative,
        x,
        tangent,
        cotangent,
        domain_pairing=pairing,
        codomain_pairing=pairing,
        abs_tol=lt2_trials.cond_dtype(5e-2, 1e-6),
    )
    
    
def test_l2t_t2l_l2t_roundtrip(rpj_dtype, rpj_batch, rpj_no_acceleration):
    lie_basis = rpj.LieBasis(4, 4)
    tensor_basis = rpj.TensorBasis(lie_basis.width, lie_basis.depth)

    x_data = rpj_batch.rng_uniform(-1, 1, lie_basis.size(), rpj_dtype)
    x = rpj.Lie(x_data, lie_basis)

    tensor_x = rpj.lie_to_tensor(x, tensor_basis)
    _ = rpj.tensor_to_lie(tensor_x, lie_basis)
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
