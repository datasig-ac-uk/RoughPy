import pytest
import numpy as np

import roughpy.compute as rpc


def test_lie_basis_creation():
    """Test basic creation of LieBasis objects"""
    width = 3
    depth = 4

    basis = rpc.LieBasis(width, depth)

    assert basis.width == width
    assert basis.depth == depth

    db = getattr(basis, "degree_begin", None)
    assert isinstance(db, np.ndarray)

    data = getattr(basis, "data", None)
    assert isinstance(data, np.ndarray)


def test_lie_basis_hash_consistency():
    """Test that lie basis objects have consistent hash values"""
    width = 3
    depth = 4

    basis1 = rpc.LieBasis(width, depth)
    basis2 = rpc.LieBasis(width, depth)

    # Same parameters should produce same hash
    assert hash(basis1) == hash(basis2)


def test_lie_basis_hash_different_width():
    """Test that lie bases with different widths have different hashes"""
    depth = 4

    basis1 = rpc.LieBasis(3, depth)
    basis2 = rpc.LieBasis(4, depth)

    # Different widths should produce different hashes
    assert hash(basis1) != hash(basis2)


def test_lie_basis_hash_different_depth():
    """Test that lie bases with different depths have different hashes"""
    width = 3

    basis1 = rpc.LieBasis(width, 3)
    basis2 = rpc.LieBasis(width, 4)

    # Different depths should produce different hashes
    assert hash(basis1) != hash(basis2)


def test_lie_basis_hash_different_params():
    """Test that lie bases with different parameters have different hashes"""
    basis1 = rpc.LieBasis(2, 3)
    basis2 = rpc.LieBasis(3, 2)
    basis3 = rpc.LieBasis(4, 5)

    # All should have different hashes
    hashes = {hash(basis1), hash(basis2), hash(basis3)}
    assert len(hashes) == 3


def test_lie_basis_hashable():
    """Test that lie basis objects are hashable and can be used in sets/dicts"""
    width = 3
    depth = 4

    basis1 = rpc.LieBasis(width, depth)
    basis2 = rpc.LieBasis(width, depth)
    basis3 = rpc.LieBasis(width + 1, depth)

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


def test_lie_basis_hash_stability():
    """Test that hash values are stable across multiple calls"""
    width = 5
    depth = 3

    basis = rpc.LieBasis(width, depth)

    # Hash should be consistent across multiple calls
    hash1 = hash(basis)
    hash2 = hash(basis)
    hash3 = hash(basis)

    assert hash1 == hash2 == hash3


def test_lie_basis_equality():
    """Test equality comparison between LieBasis objects"""
    width = 3
    depth = 4

    basis1 = rpc.LieBasis(width, depth)
    basis2 = rpc.LieBasis(width, depth)
    basis3 = rpc.LieBasis(width + 1, depth)
    basis4 = rpc.LieBasis(width, depth + 1)

    # Same parameters should be equal
    assert basis1 == basis2

    # Different parameters should not be equal
    assert basis1 != basis3
    assert basis1 != basis4
    assert basis3 != basis4


def test_lie_basis_inequality():
    """Test inequality comparison between LieBasis objects"""
    width = 3
    depth = 4

    basis1 = rpc.LieBasis(width, depth)
    basis2 = rpc.LieBasis(width, depth)
    basis3 = rpc.LieBasis(width + 1, depth)

    assert not (basis1 != basis2)

    assert basis1 != basis3



def test_lie_basis_size():
    """Test the size method of LieBasis"""
    # Test some small cases where we can verify the size
    basis1 = rpc.LieBasis(2, 1)  # Just letters
    assert basis1.size() == 2

    basis2 = rpc.LieBasis(2, 2)  # Letters + one commutator [1,2]
    assert basis2.size() == 3

    basis3 = rpc.LieBasis(3, 1)  # Three letters
    assert basis3.size() == 3


def test_lie_basis_degree_begin():
    """Test that degree_begin property works correctly"""
    width = 3
    depth = 4

    basis = rpc.LieBasis(width, depth)

    db = basis.degree_begin
    assert isinstance(db, np.ndarray)
    assert db.shape == (depth + 2,)
    assert db.dtype == np.intp

    # First entry should be 0
    assert db[0] == 0

    # Should be strictly increasing
    for i in range(1, len(db)):
        assert db[i] > db[i-1]


def test_lie_basis_data():
    """Test that data property works correctly"""
    width = 3
    depth = 3

    basis = rpc.LieBasis(width, depth)

    data = basis.data
    assert isinstance(data, np.ndarray)
    assert data.ndim == 2
    assert data.shape[1] == 2
    assert data.dtype == np.intp

    # first dim of data should be size of basis + 1
    assert data.shape[0] == basis.size() + 1

    assert data[0, 0] == 0
    assert data[0, 1] == 0


def test_lie_basis_repr():
    """Test string representation of LieBasis objects"""
    width = 5
    depth = 3

    basis = rpc.LieBasis(width, depth)
    repr_str = repr(basis)

    assert f"LieBasis({width}, {depth})" in repr_str

