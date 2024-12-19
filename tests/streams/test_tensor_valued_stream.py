
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

import roughpy as rp



@pytest.fixture
def simple_tv_stream():
    """
    Fixture to initialize a simple TensorValuedStream instance for testing.
    """
    ctx = rp.get_context(4, 2, rp.DPReal)
    increment_stream = rp.BrownianStream.with_generator("pcg64", ctx=ctx)
    initial_value = rp.FreeTensor([1.0], ctx=ctx)
    domain = rp.RealInterval(0.0, 1.0)

    return rp.TensorValuedStream(
        increment_stream=increment_stream,
        initial_value=initial_value,
        domain=domain
    )


def test_tv_stream_initial_value(simple_tv_stream):
    """
    Test the initial_value method of the TensorValuedStream class.
    """
    expected_value = rp.FreeTensor([1.0], ctx=rp.get_context(4, 2, rp.DPReal))
    assert simple_tv_stream.initial_value() == expected_value


def test_tv_stream_terminal_value(simple_tv_stream):
    """
    Test the terminal_value method of the TensorValuedStream class.
    """
    expected_value = simple_tv_stream.signature() * rp.FreeTensor([1.0], ctx=rp.get_context(4, 2, rp.DPReal))
    assert_array_almost_equal(simple_tv_stream.terminal_value(), expected_value)


def test_tv_stream_increment_stream(simple_tv_stream):
    """
    Test the increment_stream method.
    """
    stream = simple_tv_stream.increment_stream()
    assert isinstance(stream, rp.Stream)


def test_tv_stream_domain(simple_tv_stream):
    """
    Test the domain method.
    """
    domain = simple_tv_stream.domain()
    assert domain == rp.RealInterval(0.0, 1.0)


def test_tv_stream_signature(simple_tv_stream):
    """
    Test the signature method with default arguments.
    """
    ctx = rp.get_context(4, 2, rp.DPReal)
    signature = simple_tv_stream.signature()
    assert isinstance(signature, rp.FreeTensor)
    assert signature.context == ctx


def test_tv_stream_log_signature(simple_tv_stream):
    """
    Test the log_signature method with default arguments.
    """
    ctx = rp.get_context(4, 2, rp.DPReal)
    log_signature = simple_tv_stream.log_signature()
    assert isinstance(log_signature, rp.Lie)
    assert log_signature.context == ctx


def test_tv_stream_query(simple_tv_stream):
    """
    Test the query method of the TensorValuedStream class.
    """
    query_domain = rp.RealInterval(0.2, 0.8)
    result = simple_tv_stream.query(query_domain)
    assert isinstance(result, rp.TensorValuedStream)
    assert result.domain() == query_domain





def test_tv_stream_from_values():
    ctx = rp.get_context(2, 2, rp.DPReal)
    values = [
        (1.0, rp.FreeTensor([1.0, 1.0, 2.0], ctx=ctx)),
        (2.0, rp.FreeTensor([1.0, -2., 3.0], ctx=ctx)),
        (3.0, rp.FreeTensor([1., 0.5, 2.2], ctx=ctx))
    ]

    stream = rp.TensorValuedStream.from_values(values, ctx=ctx)

    assert stream.initial_value() == values[0][1]

    terminal_value = stream.terminal_value()
    first_terms = np.array(terminal_value)[:3]

    # The terminal value will have accumulated some additional higher order
    # terms because we did not restrict the value degree to 1. It should
    # be the case though, regardless of the higher order terms, that the
    # level 0 and level 1 terms should match
    assert_array_almost_equal(first_terms, np.array([1.0, 0.5, 2.2]))




def test_signature_value_stream():

    ctx = rp.get_context(4, 3, rp.DPReal)

    base_stream = rp.BrownianStream.with_generator("pcg64", ctx=ctx, support=rp.RealInterval(0., 1.))
    ## sampled Brownian stream or level1 Brownian stream

    delta_t = 0.1

    values = [
        (t, base_stream.signature(rp.RealInterval(0., t)))
        for t in np.arange(0., 1., delta_t)
    ]

    value_stream = rp.TensorValuedStream.from_values(values, ctx=ctx)
    assert value_stream.initial_value() == values[0][1]

    epsilon = 0.03125
    for i in range(len(values)-1):
        t = values[i][0]
        sub_stream = value_stream.query(rp.RealInterval(t, t+delta_t+epsilon))

        sig = sub_stream.terminal_value()
        assert_array_almost_equal(sig, values[i+1][1])
