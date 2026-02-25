"Hypothesis strategies for testing"

import jax
import jax.numpy as jnp
from hypothesis import strategies as st, settings, HealthCheck
import hypothesis.extra.numpy as hnp

from . import LieBasis, Lie, lie_to_tensor, ft_exp

# test case generation is slow, suppress warning
settings.register_profile("roughpy_jax", suppress_health_check=[HealthCheck.too_slow])
settings.load_profile("roughpy_jax")

@st.composite
def lie_increment(draw, min_width=1, max_width=6, min_depth=1, max_depth=4):
    """
    Returns a rpj.FreeTensor representing a Lie increment at a random width and depth
    """
    width = draw(st.integers(min_value=min_width, max_value=max_width))
    depth = draw(st.integers(min_value=min_depth, max_value=max_depth))
    basis = LieBasis(width, depth)
    data = jnp.array(
        draw(hnp.arrays(dtype="float32", shape=(basis.size(),), elements=st.floats(-1.0, 1.0)))
    )
    return lie_to_tensor(Lie(data, basis))  # type: ignore


@st.composite
def signature(draw, **kwargs):
    "Exponentiates Lie increments to generate signatures"
    lie = draw(lie_increment(**kwargs))
    return ft_exp(lie)