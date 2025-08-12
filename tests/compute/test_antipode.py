

import numpy as np
import pytest

from roughpy import compute
from numpy.random import default_rng
from numpy.testing import assert_array_almost_equal



def test_antipode_on_signature():

    rng = default_rng(1234)

    width = 4
    depth = 3
    basis = compute.TensorBasis(width, depth)

    x = compute.FreeTensor(np.zeros(basis.size(), dtype=np.float64), basis)

    x.data[1:1+width] = rng.normal(width)

    sig = compute.ft_exp(x)

    asig = compute.antipode(sig)

    product = compute.ft_mul(asig, sig)

    # The result should be the identity
    expected = np.zeros_like(sig.data)
    expected[0] = 1.0

    assert_array_almost_equal(product.data, expected)


