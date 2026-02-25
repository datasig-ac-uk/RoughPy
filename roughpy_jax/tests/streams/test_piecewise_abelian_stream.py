import pytest

import jax
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
                
        # Compute log signature over [0, 1]
        log_sig = self.stream.log_signature(query_interval)
        
        # Check that it equals L1 (which is l1 in this case)
        assert jnp.allclose(log_sig.data, self.l1.data, atol=1e-6)
    
    @pytest.mark.parametrize("query_interval", [
        RealInterval(0.0, 1.0, IntervalType.ClOpen),
        RealInterval(1.0, 2.0, IntervalType.ClOpen),
        RealInterval(0.5, 1.5, IntervalType.ClOpen),
        RealInterval(0.0, 2.0, IntervalType.ClOpen),
        RealInterval(-0.5, 2.5, IntervalType.ClOpen),
        RealInterval(2.0, 3.0, IntervalType.ClOpen),
        RealInterval(-2.0, -1.0, IntervalType.ClOpen),
    ])
    def test_log_signature_various_intervals(self, rpj_batch, rpj_dtype, query_interval):
        """Test log signature over various query intervals."""
        self.setup(rpj_batch, rpj_dtype)
        
        log_sig = self.stream.log_signature(query_interval) 
    
    def test_signature(self, rpj_batch, rpj_dtype):
        """Test that the signature of the stream over [0, 1] is exp(L1)."""
        self.setup(rpj_batch, rpj_dtype)
        query_interval = RealInterval(0.0, 1.0, IntervalType.ClOpen)
        sig = self.stream.signature(query_interval)
        
    # @pytest.mark.skip(reason="CBH formula not implemented yet")
    def test_log_signature_cbh(self, rpj_nobatch, rpj_dtype):
        """Test that the log signature of the stream over [0.5, 1.5] is CBH(0.5*L1, 0.5*L2)."""
        self.setup(rpj_nobatch, rpj_dtype)
        
        query_interval = RealInterval(0.5, 1.5, IntervalType.ClOpen)
        log_sig = self.stream.log_signature(query_interval)
                
        acc = self.stream._get_identity(dtype=rpj_dtype)
        
        # Compute the expected log signature using the CBH formula
        for l in (self.l1, self.l2):
            l_half = rpj.Lie(l.data * 0.5, l.basis)
            t = rpj.lie_to_tensor(l_half, tensor_basis=self.tensor_basis)
            acc = rpj.ft_fmexp(acc, t, self.tensor_basis)
                
        expected_log_sig = rpj.tensor_to_lie(rpj.ft_log(acc), lie_basis=self.lie_basis)

        assert jnp.allclose(log_sig.data, expected_log_sig.data, atol=1e-6)
    
    def test_get_identity_batch(self, rpj_batch, rpj_dtype):
        """Test that the identity element has the correct shape and properties."""
        self.setup(rpj_batch, rpj_dtype)
        
        identity = self.stream._get_identity(dtype=rpj_dtype)
        
        assert identity.data.shape[-1] == self.tensor_basis.size()
        assert identity.data.shape == (*rpj_batch.shape, self.tensor_basis.size())
        
    def test_support(self, rpj_batch, rpj_dtype):
        """Test that the support interval is correct."""
        self.setup(rpj_batch, rpj_dtype)
        
        support = self.stream.support
        
        assert support.inf == self.partition.inf
        assert support.sup == self.partition.sup
        assert support.interval_type == self.partition.interval_type
        
    def test_jitness_log_signature(self, rpj_batch, rpj_dtype):
        """Test that the log signature can be JIT compiled."""
        self.setup(rpj_batch, rpj_dtype)
        query_interval = RealInterval(0.0, 1.0, IntervalType.ClOpen)
        
        # JIT compile the log signature method, keeping the query interval as a 
        # static argument
        jit_log_sig = jax.jit(self.stream.log_signature, static_argnums=(0,))
        log_sig = jit_log_sig(query_interval)
        
        # Check that we're producing the same result as the non-JIT version
        non_jit_log_sig = self.stream.log_signature(query_interval)
        assert jnp.allclose(log_sig.data, non_jit_log_sig.data, atol=1e-6)
        # Check that the log signature equals L1 (which is l1 in this case)
        assert jnp.allclose(log_sig.data, self.l1.data, atol=1e-6)
        
        # Check that the JIT version is faster than the non-JIT version
        import time
        start_time = time.time()
        for _ in range(10):
            self.stream.log_signature(query_interval)
        non_jit_time = time.time() - start_time
        
        start_time = time.time()
        for _ in range(10):
            jit_log_sig(query_interval)
        jit_time = time.time() - start_time
        
        assert jit_time < non_jit_time
        