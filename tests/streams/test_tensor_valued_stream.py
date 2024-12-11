
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
