import pytest

import jax.numpy as jnp

import roughpy_jax as rpj
from roughpy_jax.streams import PiecewiseAbelianStream
from roughpy_jax.intervals import RealInterval, IntervalType, Partition


class TestPiecewiseAbelianStream:
    
    def setup(self, rpj_nobatch, rpj_dtype):
        # Create a simple piecewise abelian stream with two intervals
        # [0, 1] and [1, 2] with corresponding Lie elements L1 and L2
        self.interval = RealInterval(0.0, 2.0, IntervalType.ClOpen)
        self.query_interval = RealInterval(0.0, 1.0, IntervalType.ClOpen)
        self.partition = Partition([0.0, 1.0, 2.0], IntervalType.ClOpen)
        
        # Make some Lie elements for the stream (we can just use random data for this test)
        self.lie_basis = rpj.LieBasis(2, 2)
        self.tensor_basis = rpj.TensorBasis(self.lie_basis.width, self.lie_basis.depth)

        self.l1_data = rpj_nobatch.rng_uniform(-1, 1, self.lie_basis.size(), rpj_dtype)
        # self.l1_data = rpj_nobatch.rng.uniform(-1, 1, (self.lie_basis.size(),), dtype=rpj_dtype)
        self.l1 = rpj.Lie(self.l1_data, self.lie_basis)
        self.l2_data = rpj_nobatch.rng_uniform(-1, 1, self.lie_basis.size(), rpj_dtype)
        self.l2 = rpj.Lie(self.l2_data, self.lie_basis)
        
        # Create the piecewise abelian stream
        self.stream = PiecewiseAbelianStream(
            _data=(self.l1, self.l2),
            _partition=self.partition,
            _lie_basis=self.lie_basis,
            _group_basis=rpj.TensorBasis(self.lie_basis.width, self.lie_basis.depth)
        )
    
    def test_log_signature(self, rpj_nobatch, rpj_dtype):
        """Test the PiecewiseAbelianStream class.""" 
        self.setup(rpj_nobatch, rpj_dtype)
        
        print("L1:", self.l1.data)
        
        # Compute log signature over [0, 1]
        log_sig = self.stream.log_signature(self.query_interval)
        
        print("log_sig:", log_sig.data)
        # Check that it equals L1 (which is l1 in this case)
        assert jnp.allclose(log_sig.data, self.l1.data, atol=1e-6)
    
    def test_ftfmexp(self, rpj_nobatch, rpj_dtype):
        """Test that the ft_fmexp of the identity and a piece gives the piece back."""
        self.setup(rpj_nobatch, rpj_dtype)
        
        identity = self.stream._get_identity(dtype=self.l1.data.dtype)
        l1_t = rpj.lie_to_tensor(self.l1, tensor_basis=self.stream.group_basis)
        piece = rpj.ft_fmexp(identity, l1_t, self.stream.group_basis)
        
        assert jnp.allclose(rpj.tensor_to_lie(piece, self.stream.lie_basis).data, self.l1.data, atol=1e-6)
        
    def test_get_identity_batch(self, rpj_nobatch, rpj_dtype):
        """Test that the identity element has the correct shape and properties."""
        self.setup(rpj_nobatch, rpj_dtype)
        
        identity = self.stream._get_identity(dtype=self.l1.data.dtype)
        
        assert identity.data.shape[-1] == self.tensor_basis.size()
        assert identity.data.shape == (*rpj_nobatch.shape, self.tensor_basis.size())