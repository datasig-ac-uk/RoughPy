import numpy as np
import pytest
import roughpy_jax as rpj

@pytest.mark.parametrize("width,depth,expected", [
    (1, 1, [0, 1, 2]),
    (2, 1, [0, 1, 3]),
    (3, 1, [0, 1, 4]),
    (1, 2, [0, 1, 2, 3]),
    (2, 2, [0, 1, 3, 7]),
    (3, 2, [0, 1, 4, 13]),
    (1, 3, [0, 1, 2, 3, 4]),
    (2, 3, [0, 1, 3, 7, 15]),
    (3, 3, [0, 1, 4, 13, 40])
])
def test_tensor_basis_degree_begin(width, depth, expected):
    tensor_basis = rpj.TensorBasis(width=width, depth=depth)

    # Absolute check that the degree begin matches sample
    np.testing.assert_array_equal(
        tensor_basis.degree_begin,
        np.array(expected, dtype=np.int32)
    )

    # Defensive check that generated degree begin array has right size
    assert tensor_basis.degree_begin.size == depth + 2









