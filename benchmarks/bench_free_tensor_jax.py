"""
ASV benchmarks for RoughPy JAX free tensor operations.

These benchmarks focus specifically on JAX-implemented functions in roughpy_jax,
measuring core tensor algebra operations with different sizes and data types.
"""

import numpy as np
import jax.numpy as jnp

# # Ensure the parent directory is in sys.path for local imports
# upper_dir = Path(__file__).parent.parent
# sys.path.insert(0, str(upper_dir))
import os

print(os.getcwd())

try:
    import roughpy_jax as rpj
    from roughpy_jax.tests.rpy_test_common import jnp_to_np_float
except ImportError:
    raise ImportError(
        "roughpy_jax is required to run these benchmarks. "
        "Please install it via 'pip install roughpy-jax'."
    )
    

# Standard benchmark configurations
TENSOR_SIZES = [
    (2, 3),  # Small - development/testing
    (3, 4),  # Medium - typical use
    (4, 5),  # Large - stress testing  
    (2, 6),  # Deep - high-depth signatures
]

DTYPES = [jnp.float32, jnp.float64]

def _create_tensor_with_data(rng, basis, jnp_dtype):
    """
    Create a FreeTensor with non-zero data for meaningful benchmarks.
    Only the vector part (directly after scalar part) is set to ensure
    the overall value does not collapse to zero.
    """
    data = np.zeros(basis.size(), dtype=jnp_to_np_float(jnp_dtype))
    data[1:basis.width + 1] = rng.normal(size=(basis.width,))
    return rpj.FreeTensor(data, basis)

def _create_zero_tensor(basis, jnp_dtype):
    """Create a FreeTensor initialized to zero."""
    data = np.zeros(basis.size(), dtype=jnp_to_np_float(jnp_dtype))
    return rpj.FreeTensor(data, basis)

class FreeTensorBenchmarks:
    """Benchmarks for free tensor multiplication operations."""
    
    params = [TENSOR_SIZES, DTYPES]
    param_names = ['size', 'jnp_dtype']
    
    def setup(self, size, jnp_dtype):
        """Setup test tensors for each parameter combination."""
        width, depth = size
        self.basis = rpj.TensorBasis(width, depth)
        
        # Create random test data
        rng = np.random.default_rng(12345)
        size = self.basis.size()
        
        self.tensor_a = _create_tensor_with_data(rng, self.basis, jnp_dtype)
        self.tensor_b = _create_tensor_with_data(rng, self.basis, jnp_dtype)
        self.tensor_c = _create_tensor_with_data(rng, self.basis, jnp_dtype)
        
        self.signature_tensor = rpj.ft_exp(self.tensor_a)
    
    def time_ft_fma(self, size, dtype):
        """Time free tensor fused multiply-add: a + b * c"""
        return rpj.ft_fma(self.tensor_a, self.tensor_b, self.tensor_c)
    
    def time_ft_mul(self, size, dtype):
        """Time free tensor multiply: a * b"""
        return rpj.ft_mul(self.tensor_a, self.tensor_b)
    
    def time_ft_exp(self, size, dtype):
        """Time free tensor exponential: exp(a)"""
        return rpj.ft_exp(self.tensor_a)
    
    def time_ft_log(self, size, dtype):
        """Time free tensor logarithm: log(a)"""
        return rpj.ft_log(self.tensor_a)
    
    def time_ft_fmexp(self, size, dtype):
        """Time free tensor fused multiply-exponential: b * exp(a)"""
        return rpj.ft_fmexp(self.tensor_b, self.tensor_a)
    
    def time_antipode(self, size, dtype):
        """Time free tensor antipode"""
        return rpj.antipode(self.signature_tensor)

class FreeTensorZeroBenchmarks(FreeTensorBenchmarks):
    """Benchmarks for free tensor operations on zero-initialized tensors."""
    
    params = [TENSOR_SIZES, DTYPES]
    param_names = ['size', 'jnp_dtype']
    
    def setup(self, size, jnp_dtype):
        """Setup zero-initialized test tensors for each parameter combination."""
        width, depth = size
        self.basis = rpj.TensorBasis(width, depth)
        
        self.a = _create_zero_tensor(self.basis, jnp_dtype)
        self.b = _create_zero_tensor(self.basis, jnp_dtype)
        self.c = _create_zero_tensor(self.basis, jnp_dtype)
    

if __name__ == "__main__":
    import asv
    asv.run_benchmarks()

    



