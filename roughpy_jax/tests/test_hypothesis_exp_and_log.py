"Property-based tests for ft_exp and ft_log"

import jax.numpy as jnp
import roughpy_jax as rpj
from numpy.testing import assert_array_almost_equal
from hypothesis import strategies as st, given


@st.composite
def lie_increment(draw, scale=0.8):
    """
    Returns a rpj.FreeTensor representing a Lie increment at depth=2 width=2
    - level0 is 0 (Lie element)
    - level1 is two arbitrary floats (scaled by `scale`)
    - level2 is constrained to be in span{ e1⊗e2 - e2⊗e1 } i.e. (0, c, -c, 0),
      where c is drawn and scaled by `scale * scale`

    Parameters:
      scale: scale factor for coefficients, used as a decay for coeffs at a higher levels
    """
    # draw level-1 coefficients
    a = draw(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    b = draw(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    # draw single scalar for the antisymmetric level-2 part
    c_raw = draw(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )

    # apply scaling choices (helps numerical stability in tests)
    a = float(a * scale)
    b = float(b * scale)
    c = float(c_raw * scale * scale)

    basis = rpj.TensorBasis(2, 2)
    return rpj.FreeTensor(jnp.array([0, a, b, 0, c, -c, 0]), basis)


@st.composite
def signature(draw):
    "Exponentiates Lie increments to generate signatures"
    lie = draw(lie_increment())
    return rpj.ft_exp(lie)


@given(L=lie_increment())
def test_valid_lie_increment(L):
    assert L.data[0] == 0
    assert L.data[4] == -L.data[5]


@given(L=lie_increment())
def test_exp_and_log_roundtrip(L):
    a = rpj.ft_exp(L)
    expected = rpj.ft_log(a)
    assert_array_almost_equal(expected.data, L.data)
