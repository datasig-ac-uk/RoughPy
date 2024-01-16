
import roughpy as rp
from roughpy import tensor_functions


def test_adjoint_of_free_tensor():

    tensor = rp.FreeTensor([1., 2., 3., 4., 5., 6., 7.], width=2, depth=2)
    ad_tensor = tensor_functions._adjoint_of_shuffle(tensor)

    print(ad_tensor)


def test_Log_on_grouplike():
    x = rp.FreeTensor([0., 1., 2., 3.], width=3, depth=2, dtype=rp.Rational).exp()

    result = tensor_functions.Log(x)
    expected = x.log()
    assert result == expected, f"{result} != {expected}"
