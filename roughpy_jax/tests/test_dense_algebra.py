from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import roughpy_jax as rpj

from roughpy_jax.dense_algebra import (
    DenseAlgebra,
    DenseTensor,
    identity_like,
    to_dual,
    zero_like,
)


@dataclass(frozen=True)
class MockBasis:
    width: int = 5
    depth: int = 5
    degree_begin: np.ndarray = field(
        default_factory=lambda: np.array([0, 0, 1, 2, 3, 4, 5], dtype=np.int64)
    )

    def size(self) -> int:
        return 5

    def __eq__(self, other) -> bool:
        if not isinstance(other, MockBasis):
            return NotImplemented
        return (
            self.width == other.width
            and self.depth == other.depth
            and np.array_equal(self.degree_begin, other.degree_begin)
        )

    def __hash__(self) -> int:
        return hash((self.width, self.depth, tuple(self.degree_begin.tolist())))


class AlgebraA(DenseAlgebra[MockBasis]):
    pass


class AlgebraB(DenseAlgebra[MockBasis]):
    pass


AlgebraA.DualVector = AlgebraB
AlgebraB.DualVector = AlgebraA


def test_dense_algebra_rejects_scalar_data():
    basis = rpj.TensorBasis(2, 2)

    with pytest.raises(ValueError, match="at least one dimension"):
        DenseAlgebra(1.0, basis)


def test_dense_algebra_rejects_wrong_trailing_dimension():
    basis = rpj.TensorBasis(2, 2)
    data = jnp.zeros((2, basis.size() - 1), dtype=jnp.float32)

    with pytest.raises(ValueError, match="basis size must match"):
        DenseAlgebra(data, basis)


def test_dense_algebra_exposes_basic_shape_properties():
    basis = rpj.TensorBasis(2, 2)
    data = jnp.zeros((3, 4, basis.size()), dtype=jnp.float32)
    algebra = DenseAlgebra(data, basis)

    assert algebra.dtype == data.dtype
    assert algebra.shape == data.shape
    assert algebra.batch_shape == (3, 4)
    assert algebra.dimension == basis.size()


def test_dense_algebra_zero_constructs_batched_zero():
    basis = rpj.TensorBasis(2, 2)

    result = DenseAlgebra.zero(basis, dtype=jnp.float64, batch_dims=(2, 3))

    assert isinstance(result, DenseAlgebra)
    assert result.basis == basis
    assert result.data.shape == (2, 3, basis.size())
    assert result.data.dtype == jnp.float64
    assert jnp.all(result.data == 0)


def test_dense_algebra_change_depth_truncates_data():
    basis = rpj.TensorBasis(2, 3)
    data = jnp.arange(basis.size(), dtype=jnp.float32)
    algebra = DenseAlgebra(data, basis)
    new_basis = rpj.TensorBasis(2, 2)

    result = algebra.change_depth(2)

    assert result.basis == new_basis
    np.testing.assert_array_equal(result.data, data[: new_basis.size()])


def test_dense_algebra_change_depth_pads_data():
    basis = rpj.TensorBasis(2, 2)
    data = jnp.arange(basis.size(), dtype=jnp.float32)
    algebra = DenseAlgebra(data, basis)
    new_basis = rpj.TensorBasis(2, 3)

    result = algebra.change_depth(3)

    assert result.basis == new_basis
    np.testing.assert_array_equal(result.data[: basis.size()], data)
    np.testing.assert_array_equal(result.data[basis.size() :], 0)


def test_dense_algebra_change_depth_same_depth_returns_self():
    basis = rpj.TensorBasis(2, 2)
    algebra = DenseAlgebra(jnp.zeros(basis.size(), dtype=jnp.float32), basis)

    result = algebra.change_depth(2)

    assert result is algebra


@pytest.mark.parametrize("lhs_is_deeper", [False, True])
def test_dense_algebra_addition_promotes_to_deeper_basis(lhs_is_deeper):
    shallow_basis = rpj.TensorBasis(2, 2)
    deep_basis = rpj.TensorBasis(2, 3)

    if lhs_is_deeper:
        lhs = DenseTensor(jnp.ones(deep_basis.size(), dtype=jnp.float32), deep_basis)
        rhs = DenseTensor(jnp.ones(shallow_basis.size(), dtype=jnp.float32), shallow_basis)
    else:
        lhs = DenseTensor(jnp.ones(shallow_basis.size(), dtype=jnp.float32), shallow_basis)
        rhs = DenseTensor(jnp.ones(deep_basis.size(), dtype=jnp.float32), deep_basis)

    result = lhs + rhs

    assert isinstance(result, DenseTensor)
    assert result.basis == deep_basis
    expected = jnp.ones(deep_basis.size(), dtype=jnp.float32).at[: shallow_basis.size()].add(1)
    np.testing.assert_array_equal(result.data, expected)


@pytest.mark.parametrize("lhs_is_deeper", [False, True])
def test_dense_algebra_subtraction_promotes_to_deeper_basis(lhs_is_deeper):
    shallow_basis = rpj.TensorBasis(2, 2)
    deep_basis = rpj.TensorBasis(2, 3)

    if lhs_is_deeper:
        lhs = DenseTensor(2 * jnp.ones(deep_basis.size(), dtype=jnp.float32), deep_basis)
        rhs = DenseTensor(jnp.ones(shallow_basis.size(), dtype=jnp.float32), shallow_basis)
        expected = 2 * jnp.ones(deep_basis.size(), dtype=jnp.float32)
        expected = expected.at[: shallow_basis.size()].add(-1)
    else:
        lhs = DenseTensor(2 * jnp.ones(shallow_basis.size(), dtype=jnp.float32), shallow_basis)
        rhs = DenseTensor(jnp.ones(deep_basis.size(), dtype=jnp.float32), deep_basis)
        expected = (-1) * jnp.ones(deep_basis.size(), dtype=jnp.float32)
        expected = expected.at[: shallow_basis.size()].add(2)

    result = lhs - rhs

    assert isinstance(result, DenseTensor)
    assert result.basis == deep_basis
    np.testing.assert_array_equal(result.data, expected)


def test_dense_algebra_addition_rejects_mismatched_width():
    lhs = DenseTensor(
        jnp.ones(rpj.TensorBasis(2, 2).size(), dtype=jnp.float32), rpj.TensorBasis(2, 2)
    )
    rhs = DenseTensor(
        jnp.ones(rpj.TensorBasis(3, 2).size(), dtype=jnp.float32), rpj.TensorBasis(3, 2)
    )

    with pytest.raises(ValueError, match="basis widths must match"):
        _ = lhs + rhs


def test_dense_algebra_scalar_multiplication_and_division_preserve_basis():
    basis = rpj.TensorBasis(2, 2)
    tensor = DenseTensor(jnp.arange(basis.size(), dtype=jnp.float32), basis)

    multiplied = tensor * 2.0
    right_multiplied = 2.0 * tensor
    divided = tensor / 2.0

    assert multiplied.basis == basis
    assert right_multiplied.basis == basis
    assert divided.basis == basis
    np.testing.assert_array_equal(multiplied.data, tensor.data * 2.0)
    np.testing.assert_array_equal(right_multiplied.data, tensor.data * 2.0)
    np.testing.assert_allclose(divided.data, tensor.data / 2.0)


def test_dense_algebra_batched_scalar_array_multiplication_is_elementwise():
    basis = rpj.TensorBasis(2, 2)
    data = jnp.arange(2 * basis.size(), dtype=jnp.float32).reshape(2, basis.size())
    tensor = DenseTensor(data, basis)
    scalars = jnp.asarray([2.0, 3.0], dtype=jnp.float32)

    result = tensor * scalars

    assert result.basis == basis
    expected = data * scalars.reshape(2, 1)
    np.testing.assert_array_equal(result.data, expected)


def test_dense_algebra_batched_scalar_array_right_multiplication_is_elementwise():
    basis = rpj.TensorBasis(2, 2)
    data = jnp.arange(2 * basis.size(), dtype=jnp.float32).reshape(2, basis.size())
    tensor = DenseTensor(data, basis)
    scalars = jnp.asarray([2.0, 3.0], dtype=jnp.float32)

    result = scalars * tensor

    assert result.basis == basis
    expected = data * scalars.reshape(2, 1)
    np.testing.assert_array_equal(result.data, expected)


def test_dense_algebra_pytree_round_trip_preserves_basis_and_data():
    basis = rpj.TensorBasis(2, 2)
    tensor = DenseTensor(jnp.arange(basis.size(), dtype=jnp.float32), basis)

    leaves, treedef = jax.tree_util.tree_flatten(tensor)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert len(leaves) == 1
    assert isinstance(rebuilt, DenseTensor)
    assert rebuilt.basis == basis
    np.testing.assert_array_equal(rebuilt.data, tensor.data)


def test_dense_algebra_tree_flatten_returns_data_leaf_and_static_basis():
    basis = rpj.TensorBasis(2, 2)
    tensor = DenseTensor(jnp.arange(basis.size(), dtype=jnp.float32), basis)

    children, treedef = jax.tree_util.tree_flatten(tensor)

    assert len(children) == 1
    np.testing.assert_array_equal(children[0], tensor.data)
    rebuilt = jax.tree_util.tree_unflatten(treedef, children)
    assert rebuilt.basis == basis


def test_dense_algebra_tree_unflatten_rebuilds_instance():
    basis = rpj.TensorBasis(2, 2)
    data = jnp.arange(basis.size(), dtype=jnp.float32)
    tensor = DenseTensor(data, basis)

    children, treedef = jax.tree_util.tree_flatten(tensor)
    rebuilt = jax.tree_util.tree_unflatten(treedef, children)

    assert isinstance(rebuilt, DenseTensor)
    assert rebuilt.basis == basis
    np.testing.assert_array_equal(rebuilt.data, data)


def test_dense_tensor_identity_sets_unit_coordinate():
    basis = rpj.TensorBasis(2, 3)

    result = DenseTensor.identity(basis, dtype=jnp.float64, batch_dims=(2,))

    assert result.basis == basis
    assert result.batch_shape == (2,)
    assert result.data.shape == (2, basis.size())
    assert result.data.dtype == jnp.float64
    np.testing.assert_array_equal(result.data[..., 0], 1)
    np.testing.assert_array_equal(result.data[..., 1:], 0)


def test_zero_like_shape_type_and_basis():
    basis = rpj.TensorBasis(2, 2)
    data = jnp.arange(2 * 3 * basis.size(), dtype=jnp.float32).reshape(
        2, 3, basis.size()
    )
    tensor = DenseTensor(data, basis)

    result = zero_like(tensor, dtype=jnp.float64)

    assert isinstance(result, DenseTensor)
    assert result.basis == basis
    assert result.batch_shape == (2, 3)
    assert result.data.dtype == jnp.float64
    np.testing.assert_array_equal(result.data, 0)


def test_identity_like_shape_type_and_basis():
    basis = rpj.TensorBasis(2, 2)
    data = jnp.arange(2 * 3 * basis.size(), dtype=jnp.float32).reshape(
        2, 3, basis.size()
    )
    tensor = DenseTensor(data, basis)

    result = identity_like(tensor, dtype=jnp.float64)

    assert isinstance(result, DenseTensor)
    assert result.basis == basis
    assert result.batch_shape == (2, 3)
    assert result.data.dtype == jnp.float64
    np.testing.assert_array_equal(result.data[..., 0], 1)
    np.testing.assert_array_equal(result.data[..., 1:], 0)


def test_to_dual_uses_dual_vector_class():
    basis = MockBasis()
    a = AlgebraA(jnp.arange(basis.size(), dtype=jnp.float32), basis)

    b = to_dual(a)

    assert isinstance(b, AlgebraB)
    assert b.basis == basis
    np.testing.assert_array_equal(b.data, a.data)


def test_to_dual_round_trips_between_dual_mock_algebras():
    basis = MockBasis()
    b = AlgebraB(jnp.arange(basis.size(), dtype=jnp.float32), basis)

    a = to_dual(b)

    assert isinstance(a, AlgebraA)
    assert a.basis == basis
    np.testing.assert_array_equal(a.data, b.data)
