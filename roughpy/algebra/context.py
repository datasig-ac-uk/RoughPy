import roughpy.compute as rpc
from roughpy.compute import TensorBasis, LieBasis


class AlgebraContext:
    """
    AlgebraContext provides a central object for managing Lie and Tensor bases
    at a specific width and depth, along with a scalar type.
    """

    def __init__(self, width, depth, ctype, **kwargs):
        self._width = width
        self._depth = depth
        self._ctype = ctype
        self._kwargs = kwargs

        self._lie_basis = LieBasis(width, depth)
        self._tensor_basis = TensorBasis(width, depth)

    @property
    def width(self):
        return self._width

    @property
    def depth(self):
        return self._depth

    @property
    def ctype(self):
        return self._ctype

    @property
    def lie_basis(self):
        return self._lie_basis

    @property
    def tensor_basis(self):
        return self._tensor_basis

    def lie_size(self, degree=None):
        """
        Return the size of the Lie basis up to a given degree.
        If degree is None or negative, return the size up to the maximum depth.
        """
        if degree is None or degree < 0:
            degree = self._depth

        if degree > self._depth:
            raise ValueError(
                "the requested degree exceeds the maximum degree for this basis"
            )

        return int(self._lie_basis.degree_begin[degree + 1]) - 1

    def tensor_size(self, degree=None):
        """
        Return the size of the tensor basis up to a given degree.
        If degree is None or negative, return the size up to the maximum depth.
        """
        if degree is None or degree < 0:
            degree = self._depth

        if degree > self._depth:
            raise ValueError(
                "the requested degree exceeds the maximum degree for this basis"
            )

        return int(self._tensor_basis.degree_begin[degree + 1])

    def __repr__(self):
        return f"AlgebraContext(width={self._width}, depth={self._depth}, ctype={self._ctype})"

    def __str__(self):
        return repr(self)
