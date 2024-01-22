
import roughpy as rp
from roughpy import tensor_functions


def test_adjoint_of_free_tensor():

    tensor = rp.FreeTensor([1., 2., 3., 4., 5., 6., 7.], width=2, depth=2)
    ad_tensor = tensor_functions._adjoint_of_shuffle(tensor)

    key_ = tensor_functions.tensor_word_factory(tensor.context.tensor_basis)
    data = {
        (key_(), key_()): 1.0,
        (key_(), key_(1)): 2.0,
        (key_(), key_(2)): 3.0,
        (key_(), key_(1, 1)): 4.0,
        (key_(), key_(1, 2)): 6.0,
        (key_(), key_(2, 1)): 5.0,
        (key_(), key_(2, 2)): 7.0,
        (key_(1), key_()): 2.0,
        (key_(1), key_(1)): 8.0,
        (key_(1), key_(2)): 11.0,
        (key_(2), key_()): 3.0,
        (key_(2), key_(1)): 11.0,
        (key_(2), key_(2)): 14.0,
        (key_(1, 1), key_()): 4.0,
        (key_(1, 2), key_()): 6.0,
        (key_(2, 1), key_()): 5.0,
        (key_(2, 2), key_()): 7.0
    }

    for k, v in data.items():
        assert k in ad_tensor.data
        assert v == ad_tensor.data[k]
        del ad_tensor.data[k]

    assert not ad_tensor.data

def test_Log_on_grouplike():
    tensor_data = [
        0,
        1 * rp.Monomial("x1"),
        1 * rp.Monomial("x2"),
        1 * rp.Monomial("x3")
    ]
    x = rp.FreeTensor(tensor_data, width=3, depth=2,
                      dtype=rp.RationalPoly).exp()

    result = tensor_functions.Log(x)
    expected = x.log()
    assert result == expected, f"{result} != {expected}"
