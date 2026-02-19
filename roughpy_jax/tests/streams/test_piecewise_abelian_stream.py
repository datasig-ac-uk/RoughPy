import pytest

import jax.numpy as jnp

import roughpy_jax as rpj
from roughpy_jax.streams import PiecewiseAbelianStream
from roughpy_jax.intervals import RealInterval, IntervalType, Partition


class TestPiecewiseAbelianStream:
    
    def setup(self, rpj_batch, rpj_dtype):
        # Create a simple piecewise abelian stream with two intervals
        # [0, 1] and [1, 2] with corresponding Lie elements L1 and L2
        self.interval = RealInterval(0.0, 2.0, IntervalType.ClOpen)
        self.partition = Partition([0.0, 1.0, 2.0], IntervalType.ClOpen)
        
        # Make some Lie elements for the stream (we can just use random data for this test)
        self.lie_basis = rpj.LieBasis(2, 2)
        self.tensor_basis = rpj.TensorBasis(self.lie_basis.width, self.lie_basis.depth)

        self.l1_data = rpj_batch.rng_uniform(-1, 1, self.lie_basis.size(), rpj_dtype)
        # self.l1_data = rpj_batch.rng.uniform(-1, 1, (self.lie_basis.size(),), dtype=rpj_dtype)
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
        query_interval = RealInterval(0.0, 1.0, IntervalType.ClOpen)
        
        print("L1:", self.l1.data)
        
        # Compute log signature over [0, 1]
        log_sig = self.stream.log_signature(query_interval)
        
        print("log_sig:", log_sig.data)
        # Check that it equals L1 (which is l1 in this case)
        assert jnp.allclose(log_sig.data, self.l1.data, atol=1e-6)
        
    def test_signature(self, rpj_batch, rpj_dtype):
        """Test that the signature of the stream over [0, 1] is exp(L1)."""
        self.setup(rpj_batch, rpj_dtype)
        query_interval = RealInterval(0.0, 1.0, IntervalType.ClOpen)
        sig = self.stream.signature(query_interval)
        
    def test_log_signature_cbh(self, rpj_batch, rpj_dtype):
        """Test that the log signature of the stream over [0, 2] is CBH(L1, L2)."""
        self.setup(rpj_batch, rpj_dtype)
        
        query_interval = RealInterval(0.5, 1.5, IntervalType.ClOpen)
        log_sig = self.stream.log_signature(query_interval)
        
        # Compute the expected log signature using the CBH formula
        # NOTE: Use the cbh from lie_increment_stream?
        expected_log_sig = rpj.cbh(self.l1, self.l2)
        
        assert jnp.allclose(log_sig.data, expected_log_sig.data, atol=1e-6)
    
    def test_get_identity_batch(self, rpj_batch, rpj_dtype):
        """Test that the identity element has the correct shape and properties."""
        self.setup(rpj_batch, rpj_dtype)
        
        identity = self.stream._get_identity(dtype=self.l1.data.dtype)
        
        assert identity.data.shape[-1] == self.tensor_basis.size()
        assert identity.data.shape == (*rpj_batch.shape, self.tensor_basis.size())