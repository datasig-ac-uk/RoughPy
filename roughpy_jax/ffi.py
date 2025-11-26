import ctypes

try:
    import jax
except ImportError as e:
    raise ImportError("RoughPy JAX requires jax library. For install instructions please refer to https://docs.jax.dev/en/latest/installation.html") from e

try:
    # XLA functions loaded directly from .so rather than python module
    _rpy_jax_internals = ctypes.cdll.LoadLibrary("roughpy_jax/_rpy_jax_internals.so")
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
            platform="cpu"
        )
