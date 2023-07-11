import numpy as np
import pytest

from roughpy import DPReal, FreeTensor, Lie, get_context as _get_context


def get_context(width, depth):
    return _get_context(width, depth, DPReal)


def test_get_context_valid_range(width, depth):
    ctx = get_context(width, depth)

    assert ctx.width == width
    assert ctx.depth == depth


@pytest.mark.skip("Currently won't fail")
def test_get_context_out_of_bounds():
    with pytest.raises(ValueError):
        # utterly absurd values. In the future, we might support arbitrary
        # alphabets but not now.
        ctx = get_context(10000, 10000)


def test_lie_size(width, depth):
    ctx = get_context(width, depth)

    assert ctx.lie_size(1) == width


def test_tensor_size(width, depth):
    ctx = get_context(width, depth)

    assert ctx.tensor_size(1) == 1 + width


# @pytest.mark.skip("not yet implemented")
def test_make_zero_lie(width, depth):
    ctx = get_context(width, depth)
    l = ctx.zero_lie()

    assert isinstance(l, Lie)
    assert l.width == width
    assert l.max_degree == depth
    assert l.size() == 0
    assert l.degree() == 0


# @pytest.mark.skip("not yet implemented")
def test_lie_to_tensor(width, depth):
    l = Lie(np.array(range(1, width + 1), dtype=np.float64), width=width,
            depth=depth)
    ctx = get_context(width, depth)

    t = ctx.lie_to_tensor(l)

    assert isinstance(t, FreeTensor)
    assert t == FreeTensor(np.array(range(width + 1), dtype=np.float64),
                           width=width,
                           depth=depth)


# @pytest.mark.skip("not yet implemented")
def test_tensor_to_lie(width, depth):
    t = FreeTensor(np.array(range(width + 1), dtype=np.float64), width=width,
                   depth=depth)
    ctx = get_context(width, depth)

    l = ctx.tensor_to_lie(t)

    assert isinstance(l, Lie)
    assert l == Lie(np.array(range(1, width + 1), dtype=np.float64),
                    width=width,
                    depth=depth)
