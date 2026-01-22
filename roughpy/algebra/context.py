import warnings

import numpy as np

import roughpy.compute as rpc
from roughpy.compute import TensorBasis, LieBasis

DPReal = np.dtype("float64")
SPReal = np.dtype("float32")
HPReal = np.dtype("float16")


class AlgebraContext:
    """
    AlgebraContext provides a central object for managing Lie and Tensor bases
    at a specific width and depth, along with a scalar type.
    """

    width: np.int32
    depth: np.int32
    dtype: np.dtype

    lie_basis: rpc.LieBasis
    tensor_basis: rpc.TensorBasis

    def __init__(self, width, depth, dtype):
        self.width = np.int32(width)
        self.depth = np.int32(depth)
        self.dtype = np.dtype(dtype)

        self.lie_basis = LieBasis(width, depth)
        self.tensor_basis = TensorBasis(width, depth)

    @property
    def ctype(self):
        return self.dtype

    def lie_size(self, degree=None):
        """
        Return the size of the Lie basis up to a given degree.
        If degree is None or negative, return the size up to the maximum depth.
        """
        if degree is None or degree < 0:
            degree = self.depth

        if degree > self.depth:
            raise ValueError(
                "the requested degree exceeds the maximum degree for this basis"
            )

        return int(self.lie_basis.degree_begin[degree + 1]) - 1

    def tensor_size(self, degree=None):
        """
        Return the size of the tensor basis up to a given degree.
        If degree is None or negative, return the size up to the maximum depth.
        """
        if degree is None or degree < 0:
            degree = self.depth

        if degree > self.depth:
            raise ValueError(
                "the requested degree exceeds the maximum degree for this basis"
            )

        return int(self.tensor_basis.degree_begin[degree + 1])

    def __repr__(self):
        return f"AlgebraContext(width={self.width}, depth={self.depth}, ctype={self.ctype})"

    def __str__(self):
        return repr(self)


_CONTEXT_CACHE: dict[tuple[int, int, np.dtype], AlgebraContext] = {}


def _check_width_depth_dtype(width, depth, dtype):
    if width == 0 or depth == 0:
        return

    item_size = dtype.itemsize
    if width == 1:
        if depth * item_size > ((1 << 63) - 1):
            raise ValueError("resulting tensor size would be too large")

    tensor_size = (width ** (depth + 1) - 1) // (width - 1)
    tensor_bytes = tensor_size * item_size

    if tensor_bytes > ((1 << 63) - 1):
        raise ValueError("resulting tensor size would be too large")


def get_context(width: int, depth: int, dtype: np.dtype, **options) -> AlgebraContext:
    global _CONTEXT_CACHE

    if options:
        warnings.warn(
            "additional options are not supported, the following arguments are"
            f"ignored: {' '.join(options)}"
        )

    key = (width, depth, dtype)

    if (ctx := _CONTEXT_CACHE.get(key, None)) is not None:
        return ctx

    _check_width_depth_dtype(width, depth, dtype)

    ctx = _CONTEXT_CACHE[key] = AlgebraContext(width, depth, dtype)
    return ctx
