
import roughpy as rp
from roughpy import tensor_functions

import pytest




def test_adjoint_of_free_tensor():

    tensor = rp.FreeTensor([1., 2., 3., 4., 5., 6., 7.], width=2, depth=2)
    ad_tensor = tensor_functions._adjoint_of_shuffle(tensor)

    print(ad_tensor)