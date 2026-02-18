import pytest

import roughpy_jax as rpj
from roughpy_jax.streams import PiecewiseAbelianStream
from roughpy_jax.intervals import RealInterval, IntervalType, Partition


def test_piecewise_abelian_stream(rpj_batch, rpj_dtype):
    """Test the PiecewiseAbelianStream class.""" 
    # Create a piecewise abelian path
    # Give it an interval [t0, t1, t2, ..., tn]
    # Corresponding Lies [L1, L2, ..., Ln]
    # Calculate the log signature over the interval [t0, t1] and make sure it 
    # equals L1
    
    # Create a simple piecewise abelian stream with two intervals
    # [0, 1] and [1, 2] with corresponding Lie elements L1 and L2
    interval = RealInterval(0.0, 2.0, IntervalType.ClOpen)
    query_interval = RealInterval(0.0, 1.0, IntervalType.ClOpen)
    partition = Partition([0.0, 1.0, 2.0], IntervalType.ClOpen)
    
    # Make some Lie elements for the stream (we can just use random data for this test)
    lie_basis = rpj.LieBasis(2, 2)
    tensor_basis = rpj.TensorBasis(lie_basis.width, lie_basis.depth)

    l1_data = rpj_batch.rng_uniform(-1, 1, lie_basis.size(), rpj_dtype)
    l1 = rpj.Lie(l1_data, lie_basis)
    l2_data = rpj_batch.rng_uniform(-1, 1, lie_basis.size(), rpj_dtype)
    l2 = rpj.Lie(l2_data, lie_basis)
    
    # Create the piecewise abelian stream
    stream = PiecewiseAbelianStream(
        _data=(l1, l2),
        _partition=partition,
        _lie_basis=lie_basis,
        _group_basis=rpj.TensorBasis(lie_basis.width, lie_basis.depth)
    )
    
    # Compute log signature over [0, 1]
    log_sig = stream.log_signature(query_interval)
    
    # Check that it equals L1 (which is l1 in this case)
    assert log_sig == l1
    
