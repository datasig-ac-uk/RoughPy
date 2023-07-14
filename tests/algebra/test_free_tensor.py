import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

import roughpy
from roughpy import FreeTensor, TensorKey

DEPTH_LIMITS = {
    2: range(2, 26),
    3: range(2, 16),
    4: range(2, 11),
    5: range(2, 11),
}


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def rdata(rng, width):
    return rng.uniform(0.0, 1.0, size=1 + width)


@pytest.fixture
def rtensor(width, rdata):
    return FreeTensor(rdata, width=width)


@pytest.fixture
def data1(tensor_size, rng):
    return rng.uniform(0.0, 1.0, size=tensor_size)


@pytest.fixture
def data2(tensor_size, rng):
    return rng.uniform(1.0, 2.0, size=tensor_size)


def test_create_single_float_no_ctype():
    result = FreeTensor(1.0, width=2, depth=2)

    np_result = np.array(result)
    assert np_result.shape == (1,)
    assert_array_equal(np_result, np.array([1.]))
    assert result.width == 2
    assert result.max_degree == 2


def test_create_single_int_no_ctype():
    result = FreeTensor(1, width=2, depth=2)

    np_result = np.array(result)
    assert np_result.shape == (1,)
    assert_array_equal(np_result, np.array([1.]))
    assert result.width == 2
    assert result.max_degree == 2


def test_create_list_floats_no_ctype():
    result = FreeTensor([0., 1., 2., 3.], width=3, depth=2)

    np_result = np.array(result)
    assert np_result.shape == (4,)
    assert np_result.dtype == np.float64
    assert_array_equal(np_result, np.array([0., 1., 2., 3.]))
    assert result.width == 3
    assert result.max_degree == 2


def test_create_list_ints_no_ctype():
    result = FreeTensor([0, 1, 2, 3], width=3, depth=2)

    np_result = np.array(result)
    assert np_result.shape == (4,)
    assert np_result.dtype == np.float64
    assert_array_equal(np_result, np.array([0., 1., 2., 3.]))
    assert result.width == 3
    assert result.max_degree == 2


def test_create_nested_list_ints_no_ctype():
    with pytest.raises(ValueError):
        result = FreeTensor([[0, 1, 2, 3]], width=3, depth=3)


def test_create_list_width_deduction():
    result = FreeTensor([0, 1, 2, 3], depth=2)

    assert result.width == 3


class DLCreateDummy:

    def __init__(self, array):
        self.array = array

    def __dlpack__(self, stream=None):
        return self.array.__dlpack__(stream=stream)


@pytest.mark.skipif(tuple(map(int, np.__version__.split("."))) < (1, 22, 0),
                    reason="dlpack support was added in NumPy 1.22")
def test_create_dlpack():
    dummy = DLCreateDummy(np.array([0., 1., 2., 3]))
    result = FreeTensor(dummy, width=3, depth=2)

    assert result.width == 3
    assert result.max_degree == 2
    assert_array_equal(result, np.array([0., 1., 2., 3.]))


def test_create_buffer_doubles():
    from array import array
    data = array('d', [0., 1., 2., 3.])
    result = FreeTensor(data, width=3, depth=2)

    assert result.width == 3
    assert result.max_degree == 2
    assert_array_equal(result, np.array([0., 1., 2., 3.]))


def test_create_buffer_floats():
    from array import array
    data = array('f', [0., 1., 2., 3.])
    result = FreeTensor(data, width=3, depth=2)

    assert result.width == 3
    assert result.max_degree == 2
    assert_array_equal(result, np.array([0., 1., 2., 3.], dtype=np.float32))


def test_create_intv_pair():
    result = FreeTensor((1, 1.0), width=2, depth=2)

    assert result.width == 2
    assert result.max_degree == 2
    assert result.storage_type == roughpy.VectorType.SparseVector
    # assert_array_equal(result, np.array([0., 1., 0.]))


def test_create_list_intv_pairs():
    result = FreeTensor([(1, 1.0), (2, 2.0)], width=2, depth=2)

    assert result.width == 2
    assert result.max_degree == 2
    assert result.storage_type == roughpy.VectorType.SparseVector
    # assert_array_equal(result, np.array([0., 1., 2.]))


def test_create_tkey_val_pair():
    k1 = TensorKey(1, width=2, depth=2)
    result = FreeTensor((k1, 1.0), width=2, depth=2)

    assert result.width == 2
    assert result.max_degree == 2
    assert result.storage_type == roughpy.VectorType.SparseVector
    # assert result[1] == 1.0


def test_create_list_tkey_val_pairs():
    data = [
        (TensorKey(1, width=2, depth=2), 1.0),
        (TensorKey(2, width=2, depth=2), 2.0)
    ]
    result = FreeTensor(data, width=2, depth=2)

    assert result.width == 2
    assert result.max_degree == 2
    assert result.storage_type == roughpy.VectorType.SparseVector


def test_create_list_intv_pairs_dense():
    result = FreeTensor([(1, 1.0), (2, 2.0)], width=2, depth=2,
                        vector_type=roughpy.VectorType.DenseVector)

    assert result.width == 2
    assert result.max_degree == 2
    assert result.storage_type == roughpy.VectorType.DenseVector
    assert_array_equal(result, np.array([0., 1., 2.]))


def test_create_dict_args():
    result = FreeTensor({1: 1., 2: 2.}, width=2, depth=2)

    assert result.width == 2
    assert result.max_degree == 2
    assert result.storage_type == roughpy.VectorType.SparseVector


def test_create_list_dicts():
    data = [
        {1: 1., 2: 2.},
        {0: 1., 2: 1.}
    ]

    with pytest.raises(ValueError):
        result = FreeTensor(data, width=2, depth=2)


def test_create_FreeTensor_width_deduction(width, rdata):
    # Tensor of width n has n+1 elements up to degree 1
    tens = FreeTensor(rdata)

    assert tens.width == width
    assert tens.degree() == 1
    assert tens.size() == rdata.size


def test_create_FreeTensor_specified_width(rng, width):
    data = rng.uniform(0.0, 1.0, size=1 + width * (1 + width))
    tens = FreeTensor(data, width=width)

    assert tens.width == width
    assert tens.degree() == 2
    assert tens.size() == data.size


def test_create_FreeTensor_specified_width_incomplete_degree_range(rng, width):
    data = rng.uniform(1.0, 2.0, size=1 + width + 1)
    tens = FreeTensor(data, width=width, depth=2)

    assert tens.width == width
    assert tens.degree() == 2
    assert tens.size() == data.size

    atens = np.array(tens)
    assert atens.size == 1 + width * (1 + width)
    assert_array_almost_equal(tens,
                              np.concatenate((data, np.zeros(width ** 2 - 1))))


def test_FreeTensor_array_roundtrip(width, rdata, rtensor):
    assert rtensor.width == width
    assert_array_equal(rdata, np.array(rtensor))


def test_FreeTensor_repr(rng, width, depth, tensor_size):
    data = rng.uniform(0.0, 1.0, size=tensor_size)
    t = FreeTensor(data, width=width, depth=depth)

    assert repr(t) == f"FreeTensor({width=}, depth={depth}, ctype=DPReal)"


def test_FreeTensor_str():
    width = 2
    depth = 2
    t = FreeTensor(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), width=width,
                   depth=depth)
    terms = [
        "1()",
        "2(1)", "3(2)",
        "4(1,1)", "5(1,2)", "6(2,1)", "7(2,2)"
    ]
    inner = " ".join(terms)

    assert str(t) == "{ " + inner + " }"


def test_FreeTensor_addition(width, depth, data1, data2):
    t1 = FreeTensor(data1, width=width, depth=depth)
    t2 = FreeTensor(data2, width=width, depth=depth)

    expected = FreeTensor(data1 + data2, width=width, depth=depth)
    assert t1 + t2 == expected


def test_FreeTensor_subraction(width, depth, data1, data2):
    t1 = FreeTensor(data1, width=width, depth=depth)
    t2 = FreeTensor(data2, width=width, depth=depth)

    expected = FreeTensor(data1 - data2, width=width, depth=depth)

    assert t1 - t2 == expected


def test_FreeTensor_smul(width, rdata):
    t1 = FreeTensor(rdata, width=width)
    expected = FreeTensor(2.0 * rdata, width=width)

    assert t1 * 2.0 == expected


def test_FreeTensor_rdiv(width, rdata):
    t1 = FreeTensor(rdata, width=width)
    expected = FreeTensor(rdata / 2.0, width=width)

    assert t1 / 2.0 == expected


def test_FreeTensor_iadd(width, depth, data1, data2):
    t1 = FreeTensor(data1, width=width, depth=depth)
    t2 = FreeTensor(data2, width=width, depth=depth)
    expected = FreeTensor(data1 + data2, width=width, depth=depth)

    t1 += t2
    assert t1 == expected


def test_FreeTensor_isub(width, depth, data1, data2):
    t1 = FreeTensor(data1, width=width, depth=depth)
    t2 = FreeTensor(data2, width=width, depth=depth)
    expected = FreeTensor(data1 - data2, width=width, depth=depth)

    t1 -= t2
    assert t1 == expected


def test_FreeTensor_ismul(width, rdata):
    t1 = FreeTensor(rdata, width=width)
    expected = FreeTensor(2.0 * rdata, width=width)

    t1 *= 2.0
    assert t1 == expected


def test_FreeTensor_irdiv(width, rdata):
    t1 = FreeTensor(rdata, width=width)
    expected = FreeTensor(rdata / 2.0, width=width)

    t1 /= 2.0
    assert t1 == expected


def test_FreeTensor_mul():
    t1 = FreeTensor(np.array([1.0, 2.0, 3.0]), width=2, depth=2)
    t2 = FreeTensor(np.array([1.0, -1.0, -1.0]), width=2, depth=2)

    expected = FreeTensor(np.array([1.0, 1.0, 2.0, -2.0, -2.0, -3.0, -3.0]),
                          width=2, depth=2)

    assert t1 * t2 == expected


def test_FreeTensor_imul():
    t1 = FreeTensor(np.array([1.0, 2.0, 3.0]), width=2, depth=2)
    t2 = FreeTensor(np.array([1.0, -1.0, -1.0]), width=2, depth=2)

    expected = FreeTensor(np.array([1.0, 1.0, 2.0, -2.0, -2.0, -3.0, -3.0]),
                          width=2, depth=2)

    t1 *= t2
    assert t1 == expected


def test_FreeTensor_mul_single_letter(width):
    depth = 3
    t = FreeTensor(np.array([0.0, 1.0] + [0.0] * (width - 1)), width=width,
                   depth=depth)

    expected = FreeTensor(np.array(
        [1.0, 1.0] + [0.0] * (width - 1) + [0.5] + [0.0] * (width ** 2 - 1) + [
            1.0 / 6.0] + [0.0] * (width ** 3 - 1)),
                          width=width, depth=depth)

    assert t.exp() == expected


def test_FreeTensor_exp(width):
    depth = 3
    t = FreeTensor(np.array([1.0] * (1 + width)), width=width, depth=depth)

    tunit = FreeTensor(np.array([1.0]), width=width, depth=depth)

    i = float(t.max_degree)
    expected = FreeTensor(np.array([1.0]), width=width, depth=depth)
    assert expected.degree() == 0
    while i:
        expected *= (t / i)
        expected += tunit
        i -= 1.0

    assert expected.degree() == depth
    assert_array_almost_equal(t.exp(), expected)


def test_FreeTensor_fmexp(width, depth, data1, data2):
    t1 = FreeTensor(data1, width=width, depth=depth)
    data2[0] = 0.0
    t2 = FreeTensor(data2, width=width, depth=depth)

    expected = t1 * t2.exp()
    t1.fmexp(t2)
    assert_array_almost_equal(t1, expected)


def test_FreeTensor_log_exp_roundtrip(width, depth, data1):
    data1[0] = 0.0
    t = FreeTensor(data1, width=width, depth=depth)

    assert_array_almost_equal(t.exp().log(), t)


def _tensor_size(width, depth):
    s = depth
    r = 1
    while s:
        r *= width
        r += 1
        s -= 1
    return r


# def test_FreeTensor_view_into_degree(width, depth, data1):
#     t = FreeTensor(data1, width=width, depth=depth)
#
#     r0 = t.degree_array(0)
#     assert r0 == data1[0]
#
#     for d in range(1, depth+1):
#         r = t.degree_array(d)
#         assert r.shape == tuple([width]*d)
#
#         start, end = _tensor_size(width, d-1), _tensor_size(width, d)
#         assert r.size == end - start
#
#         v = data1[start:end].reshape([width]*d)
#
#         assert_array_equal(r, v)


def test_coeff_and_vec_type(width, depth, data1, coeff_type, vec_type):
    t = FreeTensor(data1, width=width, depth=depth, dtype=coeff_type,
                   vector_type=vec_type)

    assert t.storage_type == vec_type
    assert t.dtype == coeff_type


def test_antipode(width, depth, data1, coeff_type, vec_type):
    t = FreeTensor(data1, width=width, depth=depth, dtype=coeff_type,
                   vector_type=vec_type)

    result = t.antipode().antipode()
    assert result == t, f"{result} {t} {result - t}"
