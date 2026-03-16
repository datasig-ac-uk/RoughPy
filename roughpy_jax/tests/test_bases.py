from dataclasses import dataclass

import jax
import numpy as np
import pytest

from roughpy_jax import bases as rpj_bases


@dataclass(frozen=True)
class MockBasis:
    width: np.int32
    depth: np.int32
    degree_begin: np.ndarray

    def size(self) -> int:
        return int(self.degree_begin[-1])


def make_mock_basis(width: int, depth: int) -> MockBasis:
    return MockBasis(
        np.int32(width),
        np.int32(depth),
        np.array([0, 1, width + 1], dtype=np.int64),
    )


def test_check_basis_compat_accepts_same_width():
    tensor_basis = rpj_bases.TensorBasis(2, 3)
    lie_basis = rpj_bases.LieBasis(2, 5)
    mock_basis = make_mock_basis(2, 4)

    rpj_bases.check_basis_compat(tensor_basis, lie_basis, mock_basis)


def test_check_basis_compat_rejects_different_width():
    first = rpj_bases.TensorBasis(2, 3)
    second = rpj_bases.LieBasis(3, 3)

    with pytest.raises(ValueError, match="Incompatible width"):
        rpj_bases.check_basis_compat(first, second)


def test_check_basis_compat_exact_rejects_different_depth():
    first = rpj_bases.TensorBasis(2, 3)
    second = rpj_bases.TensorBasis(2, 4)

    with pytest.raises(ValueError, match="Incompatible width"):
        rpj_bases.check_basis_compat(first, second, exact=True)


def test_check_basis_compat_same_type_rejects_different_basis_type():
    tensor_basis = rpj_bases.TensorBasis(2, 3)
    lie_basis = rpj_bases.LieBasis(2, 3)

    with pytest.raises(ValueError, match="Incompatible width"):
        rpj_bases.check_basis_compat(tensor_basis, lie_basis, same_type=True)


def test_result_basis_uses_selection_strategy():
    shallow = rpj_bases.TensorBasis(2, 2)
    medium = rpj_bases.TensorBasis(2, 3)
    deep = rpj_bases.TensorBasis(2, 5)

    assert rpj_bases.result_basis(shallow, medium, deep, strategy="first") is shallow
    assert rpj_bases.result_basis(shallow, medium, deep, strategy="max_depth") is deep
    assert rpj_bases.result_basis(shallow, medium, deep, strategy="min_depth") is shallow


def test_result_basis_accepts_single_iterable():
    bases = [rpj_bases.LieBasis(2, 2), rpj_bases.LieBasis(2, 4)]

    assert rpj_bases.result_basis(bases).depth == 4


def test_result_basis_rejects_empty_input():
    with pytest.raises(ValueError, match="expected at least one basis"):
        rpj_bases.result_basis([])


def test_result_basis_rejects_unknown_strategy():
    basis = rpj_bases.TensorBasis(2, 3)

    with pytest.raises(ValueError, match="unknown basis selection strategy"):
        rpj_bases.result_basis(basis, strategy="largest")


def test_to_tensor_basis_converts_basis_like_object():
    basis_like = make_mock_basis(3, 4)

    result = rpj_bases.to_tensor_basis(basis_like)

    assert isinstance(result, rpj_bases.TensorBasis)
    assert result.width == basis_like.width
    assert result.depth == basis_like.depth


def test_to_lie_basis_converts_basis_like_object():
    basis_like = make_mock_basis(4, 3)

    result = rpj_bases.to_lie_basis(basis_like)

    assert isinstance(result, rpj_bases.LieBasis)
    assert result.width == basis_like.width
    assert result.depth == basis_like.depth


def test_basis_protocol_matches_basis_like_object():
    basis_like = make_mock_basis(2, 2)

    assert isinstance(basis_like, rpj_bases.Basis)


@pytest.mark.parametrize(
    ("basis_type", "width", "depth"),
    [
        (rpj_bases.TensorBasis, 3, 4),
        (rpj_bases.LieBasis, 2, 5),
    ],
)
def test_basis_pytree_flatten_has_no_dynamic_leaves(basis_type, width, depth):
    basis = basis_type(width, depth)

    leaves, treedef = jax.tree_util.tree_flatten(basis)

    assert leaves == []
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, basis_type)
    assert rebuilt == basis


@pytest.mark.parametrize(
    ("basis_type", "width", "depth"),
    [
        (rpj_bases.TensorBasis, 4, 3),
        (rpj_bases.LieBasis, 3, 4),
    ],
)
def test_basis_pytree_unflatten_preserves_static_basis_data(basis_type, width, depth):
    basis = basis_type(width, depth)

    leaves, aux_data = rpj_bases._basis_tree_flatten(basis)

    assert leaves == ()

    rebuilt = rpj_bases._basis_tree_unflatten(aux_data, leaves)

    assert isinstance(rebuilt, basis_type)
    assert rebuilt is basis
    assert rebuilt.width == basis.width
    assert rebuilt.depth == basis.depth
    np.testing.assert_array_equal(rebuilt.degree_begin, basis.degree_begin)
