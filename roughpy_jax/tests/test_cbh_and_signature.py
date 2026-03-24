import jax.numpy as jnp

import roughpy_jax as rpj


def _atol(dtype):
    return 1e-5 if dtype == jnp.float32 else 1e-10


def test_to_signature_matches_manual_exp(rpj_dtype, rpj_batch):
    lie_basis = rpj.LieBasis(2, 3)
    tensor_basis = rpj.to_tensor_basis(lie_basis)

    data = rpj_batch.rng_uniform(-0.2, 0.2, lie_basis.size(), rpj_dtype)
    log_signature = rpj.Lie(data, lie_basis)

    result = rpj.to_signature(log_signature)
    expected = rpj.ft_exp(rpj.lie_to_tensor(log_signature), out_basis=tensor_basis)

    assert result.basis == tensor_basis
    assert jnp.allclose(result.data, expected.data, atol=_atol(rpj_dtype))


def test_to_signature_respects_explicit_basis(rpj_dtype, rpj_batch):
    lie_basis = rpj.LieBasis(2, 3)
    output_basis = rpj.TensorBasis(lie_basis.width, 2)

    data = rpj_batch.rng_uniform(-0.2, 0.2, lie_basis.size(), rpj_dtype)
    log_signature = rpj.Lie(data, lie_basis)

    result = rpj.to_signature(log_signature, tensor_basis=output_basis)
    expected = rpj.ft_exp(rpj.lie_to_tensor(log_signature), out_basis=output_basis)

    assert result.basis == output_basis
    assert jnp.allclose(result.data, expected.data, atol=_atol(rpj_dtype))


def test_to_signature_to_log_signature_roundtrip(rpj_dtype, rpj_batch):
    lie_basis = rpj.LieBasis(2, 3)

    data = rpj_batch.rng_uniform(-0.2, 0.2, lie_basis.size(), rpj_dtype)
    log_signature = rpj.Lie(data, lie_basis)

    result = rpj.to_log_signature(rpj.to_signature(log_signature))

    assert result.basis == log_signature.basis
    assert jnp.allclose(result.data, log_signature.data, atol=_atol(rpj_dtype))


def test_cbh_single_argument_returns_input_up_to_depth_change(rpj_dtype, rpj_batch):
    lie_basis = rpj.LieBasis(2, 3)

    data = rpj_batch.rng_uniform(-0.2, 0.2, lie_basis.size(), rpj_dtype)
    x = rpj.Lie(data, lie_basis)

    result = rpj.cbh(x)

    assert result.basis == x.basis
    assert jnp.allclose(result.data, x.data, atol=_atol(rpj_dtype))


def test_cbh_matches_signature_product_for_two_inputs(rpj_dtype, rpj_batch):
    lie_basis = rpj.LieBasis(2, 3)

    x_data = rpj_batch.rng_uniform(-0.1, 0.1, lie_basis.size(), rpj_dtype)
    y_data = rpj_batch.rng_uniform(-0.1, 0.1, lie_basis.size(), rpj_dtype)
    x = rpj.Lie(x_data, lie_basis)
    y = rpj.Lie(y_data, lie_basis)

    result = rpj.cbh(x, y)

    lhs = rpj.to_signature(result)
    rhs = rpj.ft_mul(rpj.to_signature(x), rpj.to_signature(y))

    assert jnp.allclose(lhs.data, rhs.data, atol=_atol(rpj_dtype))
