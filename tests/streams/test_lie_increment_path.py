import functools
import itertools
import operator

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

import roughpy
from roughpy import FreeTensor, Lie, RealInterval
from roughpy import LieIncrementStream


def path(*args, **kwargs):
    return LieIncrementStream.from_increments(*args, **kwargs)


@pytest.fixture(params=[0, 1, 5, 10, 20, 50, 100])
def length(request):
    return request.param


@pytest.fixture
def tick_indices(length):
    return np.linspace(0.0, 1.0, length)


@pytest.fixture
def tick_data(rng, length, width):
    return rng.normal(0.0, 1.0, size=(length, width))


# @pytest.fixture
# def tick_data_w_indices(rng, tick_indices, width, length):
#     if length == 0:
#         return np.array([[]])
#
#     return np.concatenate([
#         tick_indices.reshape((length, 1)),
#         rng.normal(0.0, 1.0, size=(length, width))
#     ], axis=1)


# def test_path_width_depth_with_data(tick_data_w_indices, width, depth):
#     p = path(tick_data_w_indices, width=width, depth=depth)
#
#     assert p.width == width
#     assert p.depth == depth
#     assert isinstance(p.domain(), esig.real_interval)
#     assert p.domain() == RealInterval(-float("inf"), float("inf"))


@pytest.mark.parametrize("data",
                         [np.array([]), np.array([[]]), np.array([[], []])])
def test_path_creation_odd_data(data):
    p = path(data, width=2, depth=2)

    assert p.width == 2


# def test_path_creation_flat_length_1_path(width):
#     width + 1 because we do not supply indices
# data = np.array([1.0] * (width + 1))
# p = path(data, width=width, depth=2)
#
# assert p.width == width
# assert p.depth == 2
# needs fix for broadcasting to the correct shape in function
# assert data.shape == (width+1,)


# def test_path_creation_flat_length_1_path_no_depth(width):
#     width + 1 because we do not supply indices
# data = np.array([1.0] * (width + 1))
# p = path(data, width=width)
#
# assert p.width == width
# assert p.depth == 2
# needs fix for broadcasting to the correct shape in function
# assert data.shape == (width+1,)


# @pytest.mark.skip("Broken?")
# def test_path_restrict_tick_path(tick_data_w_indices, width):
#     p = path(tick_data_w_indices, width=width, depth=2)
#
#     if p.domain() != RealInterval(0.0, 1.0):
#         pytest.skip("Non-standard domain")
#
#     p.restrict(RealInterval(0.0, 0.5))
#
#     assert p.domain() == RealInterval(0.0, 0.5)


def test_tick_path_signature_calc_accuracy():
    data = np.array([[1.0, 2.5]])
    p = path(data, width=2, depth=2, indices=np.array([0.7]))

    r1 = p.signature(0.0, 0.8, 0.5)
    expected1 = FreeTensor(np.array([1.0]), width=2, depth=2)
    assert r1 == expected1, f"expected {expected1} but was {r1}"

    r2 = p.signature(0.0, 0.8, 0.25)
    expected2 = FreeTensor(np.array([0.0, 1.0, 2.5]), width=2, depth=2).exp()
    assert r2 == expected2, f"expected {expected2} but was {r2}"


# def test_tick_path_with_time(tick_data_w_indices, width):
#     if not tick_data_w_indices.size:
#         pytest.skip("empty array not valid data.")
#     p = path(tick_data_w_indices, depth=2, include_time=True)
#
#     assert p.width == width + 1


# def test_tick_path_with_time_no_depth(tick_data_w_indices, width):
#     if not tick_data_w_indices.size:
#         pytest.skip("empty array not valid data.")
#     p = path(tick_data_w_indices, include_time=True)
#
#     assert p.width == width + 1
#     assert p.depth == 2


@pytest.fixture(params=[2, 5, 10, 25, 100])
def t_values(request):
    return np.arange(0.0, 2.0, 2.0 / request.param) + 2.0 / request.param


@pytest.fixture
def known_path_data(t_values, width):
    t = np.arange(1.0, width + 1) * (2.0 / len(t_values))
    return np.tensordot(np.ones(len(t_values)), t, axes=0)


@pytest.fixture
def solution_signature(width, depth, tensor_size):
    letters = list(range(1, width + 1))

    def sig_func(a, b):
        rv = np.zeros(tensor_size)
        rv[0] = 1.0

        for let in letters:
            rv[let] = let * (b - a)

        idx = width
        factor = 1.0
        for d in range(2, depth + 1):
            factor /= d

            for data in itertools.product(letters, repeat=d):
                idx += 1
                rv[idx] = factor * functools.reduce(operator.mul, data, 1) * (
                        b - a) ** d
        return rv

    return sig_func


def test_tpath_known_signature_calc(width, depth, t_values, known_path_data,
                                    solution_signature):
    p = path(known_path_data, indices=t_values, width=width, depth=depth)

    expected = FreeTensor(solution_signature(0.0, 2.0), width=width,
                          depth=depth)
    assert_array_almost_equal(p.signature(0.0, 3.125), expected)


def test_tpath_known_signature_calc_with_context(width, depth, t_values,
                                                 known_path_data,
                                                 solution_signature):
    p = path(known_path_data, indices=t_values, width=width, depth=2)

    expected = FreeTensor(solution_signature(0.0, 2.0), width=width,
                          depth=depth)
    assert_array_almost_equal(
        np.array(p.signature(0.0, 3.125, depth=depth)),
        np.array(expected))


def test_tick_sig_deriv_width_3_depth_1_let_2_perturb():
    p = path(np.array([[0.2, 0.4, 0.6]]), indices=np.array([0.0]), width=3,
             depth=1)
    perturbation = Lie(np.array([0.0, 1.0, 0.0]), width=3, depth=1)
    interval = RealInterval(0.0, 1.0)

    d = p.signature_derivative(interval, perturbation)

    expected = FreeTensor(np.array([0.0, 0.0, 1.0, 0.0]), width=3, depth=1)

    assert d == expected, f"expected {expected} but got {d}"


def test_tick_sig_deriv_width_3_depth_2_let_2_perturb():
    p = path(np.array([[0.2, 0.4, 0.6]]), indices=np.array([0.0]), width=3,
             depth=2)
    perturbation = Lie(np.array([0.0, 1.0, 0.0]), width=3, depth=2)
    interval = RealInterval(0.0, 1.0)

    d = p.signature_derivative(interval, perturbation)

    expected = FreeTensor(np.array([0.0, 0.0, 1.0, 0.0,
                                    0.0, 0.1, 0.0,
                                    0.1, 0.4, 0.3,
                                    0.0, 0.3, 0.0
                                    ]), width=3, depth=2)

    assert d == expected, f"expected {expected} but got {d}"


def test_tick_sig_deriv_width_3_depth_2_let_2_perturb_with_context():
    p = path(np.array([[0.2, 0.4, 0.6]]), indices=np.array([0.0]), width=3,
             depth=2)
    perturbation = Lie(np.array([0.0, 1.0, 0.0]), width=3, depth=2)
    interval = RealInterval(0.0, 1.0)

    d = p.signature_derivative(interval, perturbation, depth=2)

    expected = FreeTensor(np.array([0.0, 0.0, 1.0, 0.0,
                                    0.0, 0.1, 0.0,
                                    0.1, 0.4, 0.3,
                                    0.0, 0.3, 0.0
                                    ]), width=3, depth=2)

    assert d == expected, f"expected {expected} but got {d}"


def test_tick_path_sig_derivative(width, depth, tick_data, tick_indices, rng):
    p = path(tick_data, indices=tick_indices, width=width, depth=depth)

    def lie():
        return Lie(rng.uniform(0.0, 5.0, size=(width,)), width=width,
                   depth=depth)

    perturbations = [
        (RealInterval(0.0, 0.3), lie()),
        (RealInterval(0.3, 0.6), lie()),
        (RealInterval(0.6, 1.1), lie()),
    ]

    result = p.signature_derivative(perturbations, 5)

    expected = FreeTensor(np.array([0.0]), width=width, depth=depth)
    for ivl, per in perturbations:
        expected *= p.signature(ivl, 5)
        expected += p.signature_derivative(ivl, per, 5)

    # assert result == expected
    assert_array_almost_equal(result, expected)


def test_lie_incr_stream_from_randints(rng):
    array = rng.integers(0, 5, size=(4, 3))

    stream = LieIncrementStream.from_increments(array, width=3, depth=2)

    sig = stream.signature(RealInterval(0.0, 5.0), 2)

    assert_array_equal(np.array(sig)[:4],
                       np.hstack([[1.0], np.sum(array, axis=0)[:]]))

def test_lie_incr_stream_from_randints_no_deduction(rng):
    array = rng.integers(0, 5, size=(4, 3))

    stream = LieIncrementStream.from_increments(array, width=3, depth=2,
                                                dtype=roughpy.DPReal)

    sig = stream.signature(RealInterval(0.0, 5.0), 2)

    assert_array_equal(np.array(sig)[:4],
                       np.hstack([[1.0], np.sum(array, axis=0)[:]]))

def test_lie_incr_stream_from_randints_transposed(rng):
    array = rng.integers(0, 5, size=(4, 3))

    stream = LieIncrementStream.from_increments(array.T, width=3, depth=2)

    sig = stream.signature(RealInterval(0.0, 5.0), 2)

    assert_array_equal(np.array(sig)[:4],
                       np.hstack([[1.0], np.sum(array, axis=0)[:]]))

def test_lie_incr_stream_from_randints_no_deduction_transposed(rng):
    array = rng.integers(0, 5, size=(4, 3))

    stream = LieIncrementStream.from_increments(array.T, width=3, depth=2,
                                                dtype=roughpy.DPReal)

    sig = stream.signature(RealInterval(0.0, 5.0), 2)

    assert_array_equal(np.array(sig)[:4],
                       np.hstack([[1.0], np.sum(array, axis=0)[:]]))


TYPE_DEDUCTION_WIDTH = 3
TYPE_DEDUCTION_DEPTH = 3
TYPE_DEDUCTION_ARGS = [
    ([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], roughpy.DPReal),
    ([[1, 2, 3], [1, 2, 3]], roughpy.DPReal),
    (np.array([[1, 2, 3], [1, 2, 3]], dtype=np.int32), roughpy.DPReal),
    (np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32), roughpy.SPReal),
    (np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float64), roughpy.DPReal),
]


@pytest.mark.parametrize('data,typ', TYPE_DEDUCTION_ARGS)
def test_ctor_type_deduction(data, typ):
    stream = LieIncrementStream.from_increments(data, width=TYPE_DEDUCTION_WIDTH, depth=TYPE_DEDUCTION_DEPTH)

    assert stream.dtype == typ





def test_log_sigature_interval_alignment_default_resolution():
    rp = roughpy
    stream = LieIncrementStream.from_increments(np.array([[0., 1., 2.],[3., 4., 5.]]), depth=2)
    zero = stream.ctx.zero_lie()

    # resolution is set dynamically
    result = stream.log_signature(rp.RealInterval(0.0, 0.1))
    expected = rp.Lie([0., 1., 2.], ctx=stream.ctx)
    assert result == expected, f"{result} != {expected}"

    for i in range(1, 10):
        result = stream.log_signature(rp.RealInterval(0.1*i, 0.1*(i+1)))
        assert result == zero, f"incr {i} expected {zero} got {result}"

def test_log_sigature_interval_alignment_set_resolution():
    rp = roughpy
    stream = LieIncrementStream.from_increments(np.array([[0., 1., 2.],[3., 4., 5.]]), depth=2, resolution=4)
    zero = stream.ctx.zero_lie()

    print(f"{stream.resolution=}")

    result = stream.log_signature(rp.RealInterval(0.0, 0.1), resolution=4)
    expected = rp.Lie([0., 1., 2.], ctx=stream.ctx)
    assert result == expected, f"{result} != {expected}"

    for i in range(1, 10):
        result = stream.log_signature(rp.RealInterval(0.1*i, 0.1*(i+1)), resolution=4)
        assert result == zero, f"{result} != {zero}"


@pytest.mark.skip("Issues on Windows with the MPIR library")
def test_signature_multiplication():
    rp = roughpy
    ctx = rp.get_context(width=3, depth=2, coeffs=rp.Rational)
    data = np.array([[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]])
    indices = np.array([0., np.pi / 4, 1.0])

    stream = LieIncrementStream.from_increments(data, indices=indices, ctx=ctx)

    left = stream.signature(rp.RealInterval(0.0, np.pi / 4))
    right = stream.signature(rp.RealInterval(np.pi / 4, 2.))
    full = stream.signature(rp.RealInterval(0.0, 2.0))

    assert left * right == full, f"{left}*{right}={left * right} != {full}"
