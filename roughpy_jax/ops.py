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
    from ._rpy_jax_internals import cpu_functions
except ImportError as e:
    _rpy_jax_internals = None
    raise ImportError(
        "RoughPy JAX CPU backend is not installed correctly"
    ) from e


for func_name, capsule in cpu_functions.items():
    jax.ffi.register_ffi_target(
        func_name,
        capsule,
        platform="cpu",
    )
