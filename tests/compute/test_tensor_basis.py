

import pytest
import numpy as np

import roughpy.compute as rpc


def test_created_degree_begins():
    width = 5
    depth = 5

    basis = rpc.TensorBasis(width, depth)

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

    basis1 = rpc.TensorBasis(width, depth)
    basis2 = rpc.TensorBasis(width, depth)

    # Same parameters should produce same hash
    assert hash(basis1) == hash(basis2)


def test_tensor_basis_hash_different_width():
    """Test that tensor bases with different widths have different hashes"""
    depth = 4

    basis1 = rpc.TensorBasis(3, depth)
    basis2 = rpc.TensorBasis(4, depth)

    # Different widths should produce different hashes
    assert hash(basis1) != hash(basis2)


def test_tensor_basis_hash_different_depth():
    """Test that tensor bases with different depths have different hashes"""
    width = 3

    basis1 = rpc.TensorBasis(width, 3)
    basis2 = rpc.TensorBasis(width, 4)

    # Different depths should produce different hashes
    assert hash(basis1) != hash(basis2)


def test_tensor_basis_hash_different_params():
    """Test that tensor bases with different parameters have different hashes"""
    basis1 = rpc.TensorBasis(2, 3)
    basis2 = rpc.TensorBasis(3, 2)
    basis3 = rpc.TensorBasis(4, 5)

    # All should have different hashes
    hashes = {hash(basis1), hash(basis2), hash(basis3)}
    assert len(hashes) == 3


def test_tensor_basis_hashable():
    """Test that tensor basis objects are hashable and can be used in sets/dicts"""
    width = 3
    depth = 4

    basis1 = rpc.TensorBasis(width, depth)
    basis2 = rpc.TensorBasis(width, depth)
    basis3 = rpc.TensorBasis(width + 1, depth)

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

    basis = rpc.TensorBasis(width, depth)

    # Hash should be consistent across multiple calls
    hash1 = hash(basis)
    hash2 = hash(basis)
    hash3 = hash(basis)

    assert hash1 == hash2 == hash3


def test_tensor_basis_hash_large_params():
    """Test hash functionality with larger parameters"""
    basis_large = rpc.TensorBasis(10, 8)
    hash_large = hash(basis_large)
    assert isinstance(hash_large, int)

    # Create another with same params
    basis_large2 = rpc.TensorBasis(10, 8)
    assert hash(basis_large) == hash(basis_large2)


def test_tensor_basis_equality_and_hash():
    """Test that equal objects have equal hashes (hash contract)"""
    width = 4
    depth = 3

    basis1 = rpc.TensorBasis(width, depth)
    basis2 = rpc.TensorBasis(width, depth)
    basis3 = rpc.TensorBasis(width + 1, depth)

    assert hash(basis1) == hash(basis2)


    # This is a probabilistic assertion - hash collisions are rare but possible
    assert hash(basis1) != hash(basis3)




class DerivedTensorBasis(rpc.TensorBasis):
    pass


def test_derived_tensor_basis_hash_equal():
    """Test that hash of derived class is the same as for derived classes"""
    width = 5
    depth = 5

    rpc_tb = rpc.TensorBasis(width, depth)
    derived_tb = DerivedTensorBasis(width, depth)

    assert hash(rpc_tb) == hash(derived_tb)