import jax.numpy as jnp
import pytest
import roughpy_jax as rpj

from derivative_testing import assert_is_linear, assert_is_derivative, assert_is_adjoint_derivative
from conftest import DerivativeTrialsHelper


# Test fixture for batch of w=4, d=4 lie tensor
@pytest.fixture(params=[jnp.float32, jnp.float64])
def lt2_trials(request):
    yield DerivativeTrialsHelper(request.param, rpj.LieBasis(4, 4), rpj.Lie)


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
    x = lt2_trials.uniform_tensor()
    y = lt2_trials.uniform_tensor()

    vals = lt2_trials.uniform_data((2,))
    alpha = float(vals[0])
    beta = float(vals[1])

    assert_is_linear(rpj.lie_to_tensor, x, y, alpha, beta)


def test_l2t_derivative(lt2_trials):
    x = lt2_trials.uniform_tensor()
    tangent = lt2_trials.uniform_tensor() * lt2_trials.cond_dtype(1e-3, 1e0)

    assert_is_derivative(
        rpj.lie_to_tensor,
        rpj.lie_to_tensor_derivative,
        x,
        tangent,
        abs_tol=lt2_trials.cond_dtype(1e-2, 1e-6)
    )


def test_l2t_adjoint_derivative(lt2_trials):
    x = lt2_trials.uniform_tensor()
    tangent = lt2_trials.uniform_tensor() * lt2_trials.cond_dtype(1e-3, 1e0)
    cotangent = lt2_trials.uniform_tensor()

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
        abs_tol=lt2_trials.cond_dtype(1e-2, 1e-6)
    )
