import numpy as np
import pytest

from roughpy import TensorKey


def tensor_size_impl(width, depth):
    s = depth
    r = 1
    while s:
        r *= width
        r += 1
        s -= 1
    return r


def test_TensorKey_str_empty(width, depth):
    key = TensorKey(width=width, depth=depth)
    assert str(key) == "()"


def test_TensorKey_str_letter(width, depth):
    key = TensorKey(1, width=width, depth=depth)
    assert str(key) == "(1)"


def test_TensorKey_str_n_letters(width, depth):
    lets = [1] * depth
    key = TensorKey(lets, width=width, depth=depth)

    assert str(key) == "(" + ",".join(map(str, lets)) + ")"


@pytest.mark.parametrize("wdth, dpth, lttrs",
                         [(w, d, np.arange(1, w + 1)) for w in range(2, 6) for d
                          in range(2, 6)])
def test_TensorKey_width_derivation(wdth, dpth, lttrs, rng):
    letters = np.concatenate([[wdth], rng.choice(lttrs, size=(dpth - 1,))])
    rng.shuffle(letters)

    key = TensorKey(letters)
    assert key.width == wdth
    assert key.degree() == dpth


def test_TensorKey_out_of_bounds_fail_single_letter(width):
    with pytest.raises(ValueError):
        key = TensorKey(width + 1, width=width)


def test_TensorKey_out_of_bounds_fail_multiple_letter(width, depth):
    with pytest.raises(ValueError):
        key = TensorKey([width + 1] + [1] * (depth - 1), width=width)


@pytest.mark.parametrize("d, data", [(d, [1] * d) for d in range(1, 6)])
def test_TensorKey_depth_derivation(d, data, width):
    key = TensorKey(data, width=width)

    assert key.degree() == d
    assert key.max_degree == d


@pytest.mark.parametrize("d, data", [(d, [1] * (d + 1)) for d in range(1, 6)])
def test_TensorKey_depth_out_of_bounds_fail(d, data, width):
    with pytest.raises(ValueError):
        key = TensorKey(data, width=width, depth=d)


def test_TensorKey_from_index(width, depth, tensor_size):
    key = TensorKey(index=tensor_size - 1, width=width, depth=depth)

    assert key == TensorKey([width] * depth, width=width, depth=depth)
