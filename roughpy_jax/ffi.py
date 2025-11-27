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


def _load_backend():
    # Directory of this file, i.e. roughpy_jax package dir
    package_dir = os.path.dirname(__file__)

    # Name must match CMake's LIBRARY_OUTPUT_NAME (_rpy_jax_internals)
    lib_name = "_rpy_jax_internals.so"

    candidate = os.path.join(package_dir, lib_name)
    if not os.path.exists(candidate):
        raise OSError(
            f"RoughPy JAX CPU backend library not found at: {candidate}"
        )

    return ctypes.CDLL(candidate)


try:
    # XLA functions loaded directly from .so rather than python module
    _rpy_jax_internals = _load_backend()
except OSError as e:
    _rpy_jax_internals = None
    raise OSError("RoughPy JAX CPU backend is not installed correctly") from e
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
            jax.ffi.pycapsule(func_ptr),
            platform="cpu",
        )
