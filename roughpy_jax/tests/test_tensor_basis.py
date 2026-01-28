"""
Reproduction of tests/compute/test_tensor_basis.py

Many of these tests are moot because rpj.TensorBasis derives from the same
_rpy_compute_internals.TensorBasis class. They are duplicated regardless to
confirm that both rpc and rpj tensor basis behave the same.
"""
import pytest
import numpy as np
import roughpy_jax as rpj

sample_width_depth_degree_begin = [
    (1, 1, [0, 1, 2]),
    (2, 1, [0, 1, 3]),
    (3, 1, [0, 1, 4]),
    (1, 2, [0, 1, 2, 3]),
    (2, 2, [0, 1, 3, 7]),
    (3, 2, [0, 1, 4, 13]),
    (1, 3, [0, 1, 2, 3, 4]),
    (2, 3, [0, 1, 3, 7, 15]),
    (3, 3, [0, 1, 4, 13, 40])
]
@pytest.mark.parametrize("width,depth,expected", sample_width_depth_degree_begin)
def test_tensor_basis_degree_begin(width, depth, expected):
    tensor_basis = rpj.TensorBasis(width=width, depth=depth)

    # Absolute check that the degree begin matches sample
    np.testing.assert_array_equal(
        tensor_basis.degree_begin,
        np.array(expected, dtype=np.int32)
    )

    # Defensive check that generated degree begin array has right size
    assert tensor_basis.degree_begin.size == depth + 2


@pytest.mark.parametrize("width,depth,expected", sample_width_depth_degree_begin)
def test_tensor_basis_size(width, depth, expected):
    # Last entry in degree_begin indicates overall size of data
    tensor_basis = rpj.TensorBasis(width=width, depth=depth)
    assert tensor_basis.size() == expected[depth + 1]


def test_created_degree_begins():
    """Test that degree_begin is created correctly"""
    width = 5
    depth = 5

    basis = rpj.TensorBasis(width, depth)
    db = getattr(basis, "degree_begin", None)
    assert isinstance(db, np.ndarray)
    assert db.shape == (depth+2, )
    assert db[0] == 0

    size = 1
    for d in range(1, depth+2):
        assert db[d] == size
        size = 1 + size * width


def test_tensor_basis_hash_consistency():
    """Test that tensor basis objects have consistent hash values"""
    width = 3
    depth = 4
    basis1 = rpj.TensorBasis(width, depth)
    basis2 = rpj.TensorBasis(width, depth)

    # Same parameters should produce same hash
    assert hash(basis1) == hash(basis2)


def test_tensor_basis_hash_different_width():
    """Test that tensor bases with different widths have different hashes"""
    depth = 4

    basis1 = rpj.TensorBasis(3, depth)
    basis2 = rpj.TensorBasis(4, depth)

    # Different widths should produce different hashes
    assert hash(basis1) != hash(basis2)


def test_tensor_basis_hash_different_depth():
    """Test that tensor bases with different depths have different hashes"""
    width = 3

    basis1 = rpj.TensorBasis(width, 3)
    basis2 = rpj.TensorBasis(width, 4)

    # Different depths should produce different hashes
    assert hash(basis1) != hash(basis2)


def test_tensor_basis_hash_different_params():
    """Test that tensor bases with different parameters have different hashes"""
    basis1 = rpj.TensorBasis(2, 3)
    basis2 = rpj.TensorBasis(3, 2)
    basis3 = rpj.TensorBasis(4, 5)

    # All should have different hashes
    hashes = {hash(basis1), hash(basis2), hash(basis3)}
    assert len(hashes) == 3


def test_tensor_basis_hashable():
    """Test that tensor basis objects are hashable and can be used in sets/dicts"""
    width = 3
    depth = 4

    basis1 = rpj.TensorBasis(width, depth)
    basis2 = rpj.TensorBasis(width, depth)
    basis3 = rpj.TensorBasis(width + 1, depth)

    # Should be able to create a set
    basis_set = {basis1, basis2, basis3}
    assert len(basis_set) == 2  # basis1 and basis2 should be considered equal

    # Should be able to use as dictionary keys
    basis_dict = {
        basis1: "first",
        basis2: "second",  # This should overwrite "first"
        basis3: "third"
    }
    assert len(basis_dict) == 2
    assert basis_dict[basis1] == "second"  # Overwritten by basis2
    assert basis_dict[basis3] == "third"


def test_tensor_basis_hash_stability():
    """Test that hash values are stable across multiple calls"""
    width = 5
    depth = 3

    basis = rpj.TensorBasis(width, depth)

    # Hash should be consistent across multiple calls
    hash1 = hash(basis)
    hash2 = hash(basis)
    hash3 = hash(basis)

    assert hash1 == hash2 == hash3


def test_tensor_basis_hash_large_params():
    """Test hash functionality with larger parameters"""
    basis_large = rpj.TensorBasis(10, 8)
    hash_large = hash(basis_large)
    assert isinstance(hash_large, int)

    # Create another with same params
    basis_large2 = rpj.TensorBasis(10, 8)
    assert hash(basis_large) == hash(basis_large2)


def test_tensor_basis_equality_and_hash():
    """Test that equal objects have equal hashes (hash contract)"""
    width = 4
    depth = 3

    basis1 = rpj.TensorBasis(width, depth)
    basis2 = rpj.TensorBasis(width, depth)
    basis3 = rpj.TensorBasis(width + 1, depth)

    assert hash(basis1) == hash(basis2)

    # This is a probabilistic assertion - hash collisions are rare but possible
    assert hash(basis1) != hash(basis3)


class DerivedTensorBasis(rpj.TensorBasis):
    pass


def test_derived_tensor_basis_hash_equal():
    """Test that hash of derived class is the same as for derived classes"""
    width = 5
    depth = 5

    rpc_tb = rpj.TensorBasis(width, depth)
    derived_tb = DerivedTensorBasis(width, depth)

    assert hash(rpc_tb) == hash(derived_tb)