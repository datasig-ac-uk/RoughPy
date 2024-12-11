import pytest

import roughpy as rp


@pytest.fixture
def simple_tv_stream():
    return rp.SimpleTensorValuedStream(increment_stream=rp.BrownianStream(),
                                       initial_value=rp.FreeTensor([1.0], ctx=rp.get_context(4, 2, rp.DPreal)),
                                       domain=rp.RealInterval(0., 1.))




def test_tv_stream_initial_value(simple_tv_stream):
    assert simple_tv_stream.initial_value() == rp.FreeTensor([1.], ctx=rp.get_context(4, 2, rp.DPreal))

