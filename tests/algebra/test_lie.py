import numpy as np
import pytest
from numpy.testing import assert_array_equal

import roughpy
from roughpy import DPReal, Lie, get_context


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def lie_size(width, depth):
    ctx = get_context(width, depth, DPReal)

    def func(d=depth):
        return ctx.lie_size(d)

    return func


@pytest.fixture
def data1(rng, width, depth, lie_size):
    return rng.uniform(0.0, 1.0, size=lie_size())


@pytest.fixture
def data2(rng, width, depth, lie_size):
    return rng.uniform(1.0, 2.0, size=lie_size())


def test_create_Lie_width_deduction(width, rng):
    l = Lie(rng.uniform(0.0, 1.0, size=width))

    assert l.width == width
    assert l.degree() == 1
    assert l.size() == width


def test_create_Lie_specified_width(rng, width):
    ctx = get_context(width, 2, DPReal)
    l = Lie(rng.uniform(0.0, 1.0, size=ctx.lie_size(2)), width=width)

    assert l.width == width
    assert l.size() == ctx.lie_size(2)
    assert l.degree() == 2


def test_lie_create_single_float_fails():
    with pytest.raises(ValueError):
        l = Lie(1.0, width=2, depth=2)


def test_lie_create_single_int_fails():
    with pytest.raises(ValueError):
        l = Lie(1, width=2, depth=2)


def test_lie_create_list_ints():
    result = Lie([1, 2, 3], width=3, depth=2)

    assert result.width == 3
    assert result.max_degree == 2
    assert result.storage_type == roughpy.VectorType.DenseVector


def test_lie_create_list_floats():
    result = Lie([1.0, 2.0, 3.0], width=3, depth=2)

    assert result.width == 3
    assert result.max_degree == 2
    assert result.storage_type == roughpy.VectorType.DenseVector


def test_Lie_array_roundtrip(width, depth, data1):
    l = Lie(data1, width=width, depth=depth)
    assert_array_equal(data1, l)


def test_Lie_repr(width, depth, data1, lie_size):
    l = Lie(data1, width=width, depth=depth)
    assert repr(l) == f"Lie({width=}, depth={depth}, ctype=DPReal)"


def test_Lie_str():
    width = depth = 2

    l = Lie(np.array([1.0, 2.0, 3.0]), width=width, depth=depth)
    terms = [
        "1(1)", "2(2)",
        "3([1,2])"
    ]
    inner = " ".join(terms)

    assert str(l) == "{ " + inner + " }"


def test_Lie_addition(width, depth, data1, data2):
    l1 = Lie(data1, width=width, depth=depth)
    l2 = Lie(data2, width=width, depth=depth)

    expected = Lie(data1 + data2, width=width, depth=depth)
    assert l1 + l2 == expected


def test_Lie_subraction(width, depth, data1, data2):
    l1 = Lie(data1, width=width, depth=depth)
    l2 = Lie(data2, width=width, depth=depth)

    expected = Lie(data1 - data2, width=width, depth=depth)
    assert l1 - l2 == expected


def test_Lie_smul(width, depth, data1):
    l1 = Lie(data1, width=width, depth=depth)

    expected = Lie(2.0 * data1, width=width, depth=depth)
    assert l1 * 2.0 == expected


def test_Lie_sdiv(width, depth, data1):
    l1 = Lie(data1, width=width, depth=depth)

    expected = Lie(data1 / 2.0, width=width, depth=depth)
    assert l1 / 2.0 == expected


def test_Lie_mul(width):
    ctx = get_context(width, 2, DPReal)
    l1 = Lie(np.array([1.0] + [0.0] * (width - 1)), width=width)
    l2 = Lie(np.array([0.0, 1.0] + [0.0] * (width - 2)), width=width)

    exp_data = np.zeros(ctx.lie_size(2))
    exp_data[width] = 1.0
    expected = Lie(exp_data, width=width)

    assert l1 * l2 == expected


def test_Lie_iadd(width, depth, data1, data2):
    l1 = Lie(data1, width=width, depth=depth)
    l2 = Lie(data2, width=width, depth=depth)

    expected = Lie(data1 + data2, width=width, depth=depth)
    l1 += l2
    assert l1 == expected


def test_Lie_isub(width, depth, data1, data2):
    l1 = Lie(data1, width=width, depth=depth)
    l2 = Lie(data2, width=width, depth=depth)

    expected = Lie(data1 - data2, width=width, depth=depth)
    l1 -= l2
    assert l1 == expected


def test_Lie_ismul(width, depth, data1):
    l1 = Lie(data1, width=width, depth=depth)

    expected = Lie(2.0 * data1, width=width, depth=depth)
    l1 *= 2.0
    assert l1 == expected


def test_Lie_isdiv(width, depth, data1):
    l1 = Lie(data1, width=width, depth=depth)

    expected = Lie(data1 / 2.0, width=width, depth=depth)
    l1 /= 2.0
    assert l1 == expected


def test_Lie_imul(width):
    ctx = get_context(width, 2, DPReal)
    l1 = Lie(np.array([1.0] + [0.0] * (width - 1)), width=width)
    l2 = Lie(np.array([0.0, 1.0] + [0.0] * (width - 2)), width=width)

    exp_data = np.zeros(ctx.lie_size(2))
    exp_data[width] = 1.0
    expected = Lie(exp_data, width=width)
    l1 *= l2
    assert l1 == expected
