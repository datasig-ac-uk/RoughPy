import numpy as np
import pytest

import roughpy as rp

TYPE_DEDUCTION_WIDTH = 3
TYPE_DEDUCTION_DEPTH = 3

TYPE_DEDUCTION_ARGS = [
    (1, rp.DPReal),
    (1.0, rp.DPReal),
    ([1], rp.DPReal),
    ([1.0], rp.DPReal),
    (np.array([1], dtype="int32"), rp.DPReal),
    (np.array([1.0], dtype="float32"), rp.SPReal),
    (np.array([1.0], dtype="float64"), rp.DPReal),
]


@pytest.mark.parametrize("data,typ", TYPE_DEDUCTION_ARGS)
def test_ft_ctor_type_deduction(data, typ):
    f = rp.ShuffleTensor(data, width=TYPE_DEDUCTION_WIDTH, depth=TYPE_DEDUCTION_DEPTH)

    assert f.dtype == typ
