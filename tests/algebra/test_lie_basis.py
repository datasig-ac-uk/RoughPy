import pytest
import roughpy as rp


def test_lie_basis_iteration():

    ctx = rp.get_context(2, 3, rp.DPReal)
    basis = ctx.lie_basis

    keys = list(basis)

    # assert len(keys) == 3
    assert [str(k) for k in keys] == ["1", "2", "[1,2]", "[1,[1,2]]", "[2,[1,2]]"]
    # assert str(keys[0]) == '1'
    # assert str(keys[1]) == '2'
    # assert str(keys[3]) == "[1,2]"