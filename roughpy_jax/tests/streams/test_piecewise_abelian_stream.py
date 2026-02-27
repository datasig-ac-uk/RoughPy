import pytest

import jax
import jax.numpy as jnp

import roughpy_jax as rpj
from roughpy_jax.streams import PiecewiseAbelianStream
from roughpy_jax.intervals import RealInterval, IntervalType, Partition
from roughpy_jax.streams.piecewise_abelian_stream import AlternativePiecewiseAbelianStream


class PASHelper:
    def __init__(self, rpj_batch, rpj_dtype):
        # Create a simple piecewise abelian stream with two intervals
        # [0, 1] and [1, 2] with corresponding Lie elements L1 and L2
        self.interval = RealInterval(0.0, 2.0, IntervalType.ClOpen)
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

    @property
    def dtype(self):
        return self.l1_data.dtype
    
    def batch_shape(self):
        return self.l1_data.shape[:-1]


@pytest.fixture
def pas_data(rpj_nobatch, rpj_dtype=jnp.float32):
    return PASHelper(rpj_nobatch, rpj_dtype)


class TestPiecewiseAbelianStream:
        
    def test_construction(self, pas_data):
        """Test that the PiecewiseAbelianStream can be constructed without errors."""
        with pytest.raises(ValueError):
            PiecewiseAbelianStream(
                _data=(pas_data.l1,),  # Incorrect length of data
                _partition=pas_data.partition,
                _lie_basis=pas_data.lie_basis,
                _group_basis=rpj.TensorBasis(pas_data.lie_basis.width, pas_data.lie_basis.depth)
            )
    
    def test_log_signature(self, pas_data):
        """Test the PiecewiseAbelianStream class.""" 
        query_interval = RealInterval(0.0, 1.0, IntervalType.ClOpen)
                
        # Compute log signature over [0, 1]
        log_sig = pas_data.stream.log_signature(query_interval)
        
        # Check that it equals L1 (which is l1 in this case)
        assert jnp.allclose(log_sig.data, pas_data.l1.data, atol=1e-6)
    
    @pytest.mark.parametrize("query_interval", [
        RealInterval(0.0, 1.0, IntervalType.ClOpen),
        RealInterval(1.0, 2.0, IntervalType.ClOpen),
        RealInterval(0.5, 1.5, IntervalType.ClOpen),
        RealInterval(0.0, 2.0, IntervalType.ClOpen),
        RealInterval(-0.5, 2.5, IntervalType.ClOpen),
        RealInterval(2.0, 3.0, IntervalType.ClOpen),
        RealInterval(-2.0, -1.0, IntervalType.ClOpen),
    ])
    def test_log_signature_various_intervals(self, pas_data, query_interval):
        """Test log signature over various query intervals."""
        log_sig = pas_data.stream.log_signature(query_interval) 
    
    def test_signature(self, pas_data):
        """Test that the signature of the stream over [0, 1] is exp(L1)."""
        query_interval = RealInterval(0.0, 1.0, IntervalType.ClOpen)
        sig = pas_data.stream.signature(query_interval)
        
    def test_log_signature_cbh(self, pas_data):
        """Test that the log signature of the stream over [0.5, 1.5] is CBH(0.5*L1, 0.5*L2)."""
        
        query_interval = RealInterval(0.5, 1.5, IntervalType.ClOpen)
        log_sig = pas_data.stream.log_signature(query_interval)
                
        acc = pas_data.stream._get_identity(dtype=pas_data.dtype)
        
        # Compute the expected log signature using the CBH formula
        for l in (pas_data.l1, pas_data.l2):
            l_half = rpj.Lie(l.data * 0.5, l.basis)
            t = rpj.lie_to_tensor(l_half, tensor_basis=pas_data.tensor_basis)
            acc = rpj.ft_fmexp(acc, t, pas_data.tensor_basis)
                
        expected_log_sig = rpj.tensor_to_lie(rpj.ft_log(acc), lie_basis=pas_data.lie_basis)

        assert jnp.allclose(log_sig.data, expected_log_sig.data, atol=1e-6)
    
    def test_get_identity_batch(self, pas_data):
        """Test that the identity element has the correct shape and properties."""
        identity = pas_data.stream._get_identity(dtype=pas_data.dtype)
        
        assert identity.data.shape[-1] == pas_data.tensor_basis.size()
        assert identity.data.shape == (*pas_data.batch_shape(), pas_data.tensor_basis.size())
        
    def test_support(self, pas_data):
        """Test that the support interval is correct."""        
        support = pas_data.stream.support
        
        assert support.inf == pas_data.partition.inf
        assert support.sup == pas_data.partition.sup
        assert support.interval_type == pas_data.partition.interval_type
    
    @pytest.mark.parametrize("static", [True, False])
    def test_jitness_log_signature(self, pas_data, static):
        """Test that the log signature can be JIT compiled."""
        query_interval = RealInterval(0.0, 1.0, IntervalType.ClOpen)
        
        # JIT compile the log signature method, keeping the query interval as a 
        # static argument
        jit_log_sig = jax.jit(pas_data.stream.log_signature, static_argnums=(0,) if static else ())
        log_sig = jit_log_sig(query_interval)
        
        # Check that we're producing the same result as the non-JIT version
        non_jit_log_sig = pas_data.stream.log_signature(query_interval)
        assert jnp.allclose(log_sig.data, non_jit_log_sig.data, atol=1e-6)
        # Check that the log signature equals L1 
        assert jnp.allclose(log_sig.data, pas_data.l1.data, atol=1e-6)
        
        # Check that the JIT version is faster than the non-JIT version
        import time
        start_time = time.time()
        for _ in range(10):
            pas_data.stream.log_signature(query_interval)
        non_jit_time = time.time() - start_time
        
        start_time = time.time()
        for _ in range(10):
            jit_log_sig(query_interval)
        jit_time = time.time() - start_time
        
        assert jit_time < non_jit_time
        
        
class TestPiecewiseAbelianStreamBench():
    def test_log_signature_bench(self, benchmark, pas_data):
        """Benchmark the log signature computation."""
        query_interval = RealInterval(0.0, 1.0, IntervalType.ClOpen)
        
        benchmark(pas_data.stream.log_signature, query_interval)
        
    def test_log_signature_jit_static_bench(self, benchmark, pas_data):
        """Benchmark the JIT-compiled log signature computation."""
        query_interval = RealInterval(0.0, 1.0, IntervalType.ClOpen)
        
        jit_log_sig = jax.jit(pas_data.stream.log_signature, static_argnums=(0,))
        
        benchmark(jit_log_sig, query_interval)
    
    def test_log_signature_jit_dynamic_bench(self, benchmark, pas_data):
        """Benchmark the JIT-compiled log signature computation with dynamic query interval."""
        query_interval = RealInterval(0.0, 1.0, IntervalType.ClOpen)
        
        jit_log_sig = jax.jit(pas_data.stream.log_signature, static_argnums=())
        
        benchmark(jit_log_sig, query_interval)
        
    def test_log_signature_alternative_bench(self, benchmark, pas_data):
        """Benchmark the alternative log signature computation."""
        query_interval = RealInterval(0.0, 1.0, IntervalType.ClOpen)
        
        alt_stream = AlternativePiecewiseAbelianStream(
            _data=pas_data.stream._data,
            _partition=pas_data.stream._partition,
            _lie_basis=pas_data.stream._lie_basis,
            _group_basis=pas_data.stream._group_basis
        )
        
        benchmark(alt_stream.log_signature, query_interval)
        
    def test_log_signature_alternative_jit_bench(self, benchmark, pas_data):
        """Benchmark the JIT-compiled alternative log signature computation."""
        query_interval = RealInterval(0.0, 1.0, IntervalType.ClOpen)
        
        alt_stream = AlternativePiecewiseAbelianStream(
            _data=pas_data.stream._data,
            _partition=pas_data.stream._partition,
            _lie_basis=pas_data.stream._lie_basis,
            _group_basis=pas_data.stream._group_basis
        )
        
        jit_alt_log_sig = jax.jit(alt_stream.log_signature, static_argnums=(0,))
        
        benchmark(jit_alt_log_sig, query_interval)