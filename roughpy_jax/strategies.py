"Hypothesis strategies for testing"

import math
import jax.numpy as jnp
from hypothesis import strategies as st, settings, HealthCheck
import hypothesis.extra.numpy as hnp

from . import TensorBasis, LieBasis, Lie, lie_to_tensor, ft_exp
from .intervals import RealInterval, IntervalType

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
    tensor_basis = TensorBasis(width, depth)
    data = jnp.array(
        draw(
            hnp.arrays(
                dtype="float32", shape=(basis.size(),), elements=st.floats(-1.0, 1.0)
            )
        )
    )
    tensor = lie_to_tensor(Lie(data, basis), tensor_basis=tensor_basis)  # type: ignore
    # workaround: patch basis to use TensorBasis, instead of LieBasis
    tensor.basis = tensor_basis
    return tensor


@st.composite
def signature(draw, **kwargs):
    "Exponentiates Lie increments to generate signatures"
    lie = draw(lie_increment(**kwargs))
    return ft_exp(lie)


def strictly_increasing_float_pair():
    """Returns a pair of floats a, b, with b > a"""
    base = st.floats(allow_nan=False, allow_infinity=False)
    # filter out values where nextafter would be infinite (i.e., largest representable float)
    base = base.filter(lambda a: not math.isinf(math.nextafter(a, math.inf)))
    return base.flatmap(
        lambda a: st.tuples(
            st.just(a),
            st.floats(
                min_value=math.nextafter(a, math.inf),
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )


@st.composite
def intervals(draw, type: IntervalType):
    a, b = draw(strictly_increasing_float_pair())
    return RealInterval(a, b, _interval_type=type)
