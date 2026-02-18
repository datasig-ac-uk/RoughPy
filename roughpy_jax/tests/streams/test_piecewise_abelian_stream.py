import pytest

import roughpy_jax as rpj
from roughpy_jax.streams import PiecewiseAbelianStream
from roughpy_jax.intervals import RealInterval, IntervalType, Partition


class TestPiecewiseAbelianStream:
    
    def setup(self, rpj_batch, rpj_dtype):
        # Create a simple piecewise abelian stream with two intervals
        # [0, 1] and [1, 2] with corresponding Lie elements L1 and L2
        self.interval = RealInterval(0.0, 2.0, IntervalType.ClOpen)
        self.query_interval = RealInterval(0.0, 1.0, IntervalType.ClOpen)
        self.partition = Partition([0.0, 1.0, 2.0], IntervalType.ClOpen)
        
        # Make some Lie elements for the stream (we can just use random data for this test)
        self.lie_basis = rpj.LieBasis(2, 2)
        self.tensor_basis = rpj.TensorBasis(self.lie_basis.width, self.lie_basis.depth)

        self.l1_data = rpj_batch.rng_uniform(-1, 1, self.lie_basis.size(), rpj_dtype)
        self.l1 = rpj.Lie(self.l1_data, self.lie_basis)
        self.l2_data = rpj_batch.rng_uniform(-1, 1, self.lie_basis.size(), rpj_dtype)
        self.l2 = rpj.Lie(self.l2_data, self.lie_basis)
        
        # Create the piecewise abelian stream
        self.stream = PiecewiseAbelianStream(
            _data=(self.l1, self.l2),
            _partition=self.partition,
            _lie_basis=self.lie_basis,
            _group_basis=rpj.TensorBasis(self.lie_basis.width, self.lie_basis.depth)
        )
    
    def test_log_signature(self, rpj_batch, rpj_dtype):
        """Test the PiecewiseAbelianStream class.""" 
        self.setup(rpj_batch, rpj_dtype)
        
        # Compute log signature over [0, 1]
        log_sig = self.stream.log_signature(self.query_interval)
        
        # Check that it equals L1 (which is l1 in this case)
        assert log_sig == self.l1
        
    def test_get_identity_batch(self, rpj_batch, rpj_dtype):
        """Test that the identity element has the correct shape and properties."""
        self.setup(rpj_batch, rpj_dtype)
        
        identity = self.stream._get_identity(dtype=self.l1.data.dtype)
        
        assert identity.data.shape[-1] == self.tensor_basis.size()
        assert identity.data.shape == (*rpj_batch.shape, self.tensor_basis.size())