


import pytest
import roughpy as rp




@pytest.fixture
def borrowed_scalar():
    tensor = rp.FreeTensor([1.0, 2.0, 3.0], width=2, depth=1)
    yield tensor[rp.TensorKey(width=2,depth=1)]




def test_multiply_borrowed_scalar_by_int(borrowed_scalar):
    result = borrowed_scalar * 2

    assert result == 2.0