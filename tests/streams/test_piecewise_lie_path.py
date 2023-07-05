import pytest
from numpy.testing import assert_array_almost_equal

from roughpy import DPReal, Lie, PiecewiseAbelianStream, RealInterval, \
    VectorType, get_context

# skip = True
skip = False

WIDTH = 5
DEPTH = 3


@pytest.fixture(params=[1, 2, 5])
def count(request):
    return request.param


@pytest.fixture
def piecewise_intervals(count):
    return [RealInterval(float(i), float(i + 1)) for i in range(count)]


@pytest.fixture
def piecewise_lie_data(piecewise_intervals, rng):
    return [
        (interval,
         Lie(rng.normal(0.0, 1.0, size=(WIDTH,)), width=WIDTH, depth=DEPTH))
        for interval in piecewise_intervals
    ]


@pytest.fixture
def piecewise_lie(piecewise_lie_data):
    return PiecewiseAbelianStream.construct(piecewise_lie_data, width=WIDTH,
                                            depth=DEPTH)


@pytest.mark.skipif(skip, reason="path type not available")
def test_log_signature_full_data(piecewise_lie_data):
    ctx = get_context(WIDTH, DEPTH, DPReal)
    piecewise_lie = PiecewiseAbelianStream.construct(piecewise_lie_data,
                                                     width=WIDTH, depth=DEPTH,
                                                     dtype=DPReal)

    result = piecewise_lie.log_signature(5)
    expected = ctx.cbh([d[1] for d in piecewise_lie_data],
                       vec_type=VectorType.DenseVector)
    # assert result == expected, f"{expected}\n{result}"
    assert_array_almost_equal(expected, result)
