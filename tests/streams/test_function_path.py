import itertools
import functools
import operator

import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal

skip = True
# from roughpy import RealInterval, FreeTensor, Lie, Stream, DenseFunctionPath

def path(*args, **kwargs):
    return Stream(*args, **kwargs, type=DenseFunctionPath)

@pytest.mark.skipif(skip, reason="path type not available")
def test_function_path_signature_calc_accuracy():
    def func(t):
        if t >= 0.5:
            return np.array([t - 0.5, t - 0.5])
        return np.array([0.0, 0.0])

    p = path(func, width=2, depth=2)

    r1 = p.signature(0.0, 0.8, 0.5)
    expected1 = FreeTensor(np.array([1.0]), width=2, depth=2)
    assert r1 == expected1

    r2 = p.signature(0.0, 0.8, 0.25)  # [0, 0.5), [0.5, 0.75)
    expected2 = FreeTensor(np.array([0.0, *func(0.75)]), width=2, depth=2).exp()
    assert r2 == expected2, f"{r2} {expected2}"


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
                rv[idx] = factor * functools.reduce(operator.mul, data, 1) * (b - a) ** d
        return rv

    return sig_func


@pytest.mark.skipif(skip, reason="path type not available")
def test_fpath_known_signature_calc(width, depth, solution_signature):
    p = path(lambda t: np.arange(1.0, width + 1) * t, width=width, depth=depth)

    expected = FreeTensor(solution_signature(0.0, 2.0), width=width, depth=depth)
    assert_array_almost_equal(p.signature(0.0, 2.0, 0.0), expected)


@pytest.fixture
def deriv_function_path():
    def func(t):
        if t > 0.0:
            return np.array([0.2, 0.4, 0.6])
        return np.array([0.0, 0.0, 0.0])

    return func


@pytest.mark.skipif(skip, reason="path type not available")
def test_func_sig_deriv_s_width_3_depth_1_let_2_perturb(deriv_function_path):
    p = path(deriv_function_path, width=3, depth=1)
    perturbation = Lie(np.array([0.0, 1.0, 0.0]), width=3, depth=1)
    interval = RealInterval(0.0, 1.0)

    d = p.signature_derivative(interval, perturbation, 0.5)

    expected = FreeTensor(np.array([0.0, 0.0, 1.0, 0.0]), width=3, depth=1)

    assert d == expected


@pytest.mark.skipif(skip, reason="path type not available")
def test_func_sig_deriv_s_width_3_depth_2_let_2_perturb(deriv_function_path):
    p = path(deriv_function_path, width=3, depth=2)
    perturbation = Lie(np.array([0.0, 1.0, 0.0]), width=3, depth=2)
    interval = RealInterval(0.0, 1.0)

    d = p.signature_derivative(interval, perturbation, 0.5)

    expected = FreeTensor(np.array([0.0, 0.0, 1.0, 0.0,
                                  0.0, 0.1, 0.0,
                                  0.1, 0.4, 0.3,
                                  0.0, 0.3, 0.0
                                  ]), width=3, depth=2)

    assert_array_almost_equal(p.log_signature(interval, 0.5), np.array([0.2, 0.4, 0.6]))
    assert_array_almost_equal(d, expected)


@pytest.mark.skipif(skip, reason="path type not available")
def test_func_sig_deriv_m_width_3_depth_1_let_2_perturb(deriv_function_path):
    p = path(deriv_function_path, width=3, depth=1)
    perturbation = Lie(np.array([0.0, 1.0, 0.0]), width=3, depth=1)
    interval = RealInterval(0.0, 1.0)

    d = p.signature_derivative([(interval, perturbation)], 0.5)

    expected = FreeTensor(np.array([0.0, 0.0, 1.0, 0.0]), width=3, depth=1)

    assert_array_almost_equal(d, expected)
    # assert d == expected


@pytest.mark.skipif(skip, reason="path type not available")
def test_func_sig_deriv_m_width_3_depth_2_let_2_perturb(deriv_function_path):
    p = path(deriv_function_path, width=3, depth=2)
    perturbation = Lie(np.array([0.0, 1.0, 0.0]), width=3, depth=2)
    interval = RealInterval(0.0, 1.0)

    d = p.signature_derivative([(interval, perturbation)], 0.5)

    expected = FreeTensor(np.array([0.0, 0.0, 1.0, 0.0,
                                  0.0, 0.1, 0.0,
                                  0.1, 0.4, 0.3,
                                  0.0, 0.3, 0.0
                                  ]), width=3, depth=2)

    assert_array_almost_equal(d, expected)
    # assert d == expected
