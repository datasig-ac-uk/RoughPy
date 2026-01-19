"""
ASV benchmarks for RoughPy JAX free tensor operations.

These benchmarks focus specifically on JAX-implemented functions in roughpy_jax,
measuring core tensor algebra operations with different sizes and data types.
"""

import numpy as np
import jax.numpy as jnp

try:
    import roughpy_jax as rpj
except ImportError:
    raise ImportError(
        "roughpy_jax is required to run these benchmarks. "
        "Please install it via 'pip install roughpy_jax'."
    )


# FIXME test benchmark to check if conversion to np dtype still required
def _jnp_to_np_float(jnp_dtype):
    """
    Unit test jnp to numpy float type conversion
    """
    if jnp_dtype == jnp.float32:
        return np.float32

    if jnp_dtype == jnp.float64:
        return np.float64

    raise ValueError("Expecting f32 or f64")


# Standard benchmark configurations
TENSOR_SIZES = [
    (2, 3),  # Small - development/testing
    (3, 4),  # Medium - typical use
    (4, 5),  # Large - stress testing  
    (4, 10),  # Deep - high-depth signatures
]

DTYPES = [jnp.float32, jnp.float64]

def _create_tensor_with_data(rng, basis, jnp_dtype):
    """
    Create a FreeTensor with non-zero data for meaningful benchmarks.
    Only the vector part (directly after scalar part) is set to ensure
    the overall value does not collapse to zero.
    """
    data = np.zeros(basis.size(), dtype=_jnp_to_np_float(jnp_dtype))
    data[1:basis.width + 1] = rng.normal(size=(basis.width,))
    j_data = jnp.array(data, dtype=jnp_dtype)
    return rpj.FreeTensor(j_data, basis)

def _create_zero_tensor(basis, jnp_dtype):
    """Create a FreeTensor initialized to zero."""
    data = np.zeros(basis.size(), dtype=_jnp_to_np_float(jnp_dtype))
    j_data = jnp.array(data, dtype=jnp_dtype)
    return rpj.FreeTensor(j_data, basis)

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
        
        self.tensor_a = _create_zero_tensor(self.basis, jnp_dtype)
        self.tensor_b = _create_zero_tensor(self.basis, jnp_dtype)
        self.tensor_c = _create_zero_tensor(self.basis, jnp_dtype)
        
        self.signature_tensor = rpj.ft_exp(self.tensor_a)
    

if __name__ == "__main__":
    import asv
    asv.run_benchmarks()

    



