import ctypes
import os
from importlib import resources

try:
    import jax
except ImportError as e:
    raise ImportError(
        "RoughPy JAX requires jax library. "
        "For install instructions please refer to "
        "https://docs.jax.dev/en/latest/installation.html"
    ) from e


try:
    # Import using python module syntax to ensure proper loading of shared library
    from . import _rpy_jax_internals
except ImportError as e:
    _rpy_jax_internals = None
    raise ImportError(
        "RoughPy JAX CPU backend is not installed correctly"
    ) from e
else:
    # Register CPU functions by looking up expected names in .so
    cpu_func_names = [
        "cpu_dense_ft_fma",
        "cpu_dense_ft_exp",
        "cpu_dense_ft_log",
        "cpu_dense_ft_fmexp",
        "cpu_dense_ft_antipode",
        "cpu_dense_st_fma",
    ]
    for func_name in cpu_func_names:
        func_ptr = getattr(_rpy_jax_internals, func_name)
        jax.ffi.register_ffi_target(
            func_name,
            func_ptr,
            platform="cpu",
        )
