from typing import TypeVar, Callable, Generic, ClassVar

import jax
import jax.numpy as jnp
import numpy as np

# from roughpy_jax.bases import BasisT
BasisT = TypeVar("BasisT")  ## TODO: replace when basis utilities is merged

AlgebraT = TypeVar("AlgebraT")
_T = TypeVar("_T")


def get_common_batch_shape(*operands) -> tuple[int, ...]:
    """
    Validate that all operands share the same batch shape and return it.

    For dense algebra objects, the batch shape is the prefix of the data shape
    preceding the trailing algebra dimension. At present, operations in
    roughpy-jax require all operands to have identical batch shape, and this
    helper returns that common value after validation.

    :param operands: Dense algebra operands to validate.
    :return: The batch shape common to all operands.
    :raises ValueError: If no operands are supplied or any operand has a
        different batch shape.
    """
    if not operands:
        raise ValueError("expected at least one operand")

    first, *rem = operands
    batch_shape = first.batch_shape

    for i, operand in enumerate(rem, start=1):
        if operand.batch_shape != batch_shape:
            raise ValueError(
                f"incompatible batch shape in argument at index {i}:"
                f" expected {batch_shape} but got {operand.batch_shape}"
            )

    return batch_shape


def _pad_final_dim(data: jax.Array, size: int) -> jax.Array:
    """
    Pad the trailing algebra dimension of ``data`` with zeros up to ``size``.

    This is used when combining dense algebra elements of different truncation
    depths by embedding the shallower coefficient array into the deeper basis.

    :param data: Coefficient array to pad.
    :param size: Target size of the trailing algebra dimension.
    :return: ``data`` with zeros appended along the final dimension.
    """
    pad_width = [(0, 0)] * (len(data.shape) - 1) + [(0, size - data.shape[-1])]
    return jnp.pad(data, pad_width)


def _algebra_add(
    a: AlgebraT, b: AlgebraT, *, impl: Callable[[jax.Array, ...], jax.Array]
) -> AlgebraT:
    """
    Apply a pointwise binary operation to two compatible dense algebra objects.

    Addition and subtraction between dense algebra elements are implemented by
    first checking width and batch-shape compatibility, then promoting the
    shallower operand to the deeper basis by zero-padding its trailing algebra
    dimension.

    :param a: Left operand.
    :param b: Right operand.
    :param impl: Elementwise array implementation such as ``jnp.add``.
    :return: Result of applying ``impl`` in the deeper of the two bases.
    """
    cls = type(a)

    if not issubclass(type(b), cls):
        return NotImplemented

    if a.basis.width != b.basis.width:
        raise ValueError("basis widths must match for addition")

    get_common_batch_shape(a, b)

    if a.basis.depth >= b.basis.depth:
        result_basis = a.basis
        a_data = a.data
        b_data = _pad_final_dim(b.data, a.basis.size())
    else:
        result_basis = b.basis
        a_data = _pad_final_dim(a.data, b.basis.size())
        b_data = b.data

    result_data = impl(a_data, b_data)
    return cls(result_data, result_basis)


def _algebra_scalar_multiply(a: AlgebraT, s: jax.typing.ArrayLike) -> AlgebraT:
    """
    Multiply a dense algebra element by a scalar-like value.

    :param a: Dense algebra operand.
    :param s: Scalar-like multiplier.
    :return: Scaled algebra element in the same basis as ``a``.
    """
    cls = type(a)
    result_data = jnp.dot(a.data, s)
    return cls(result_data, a.basis)


def _redepth_data(data: jax.Array, new_alg_dim: int) -> jax.Array: ...


@jax.tree_util.register_pytree_node_class
class DenseAlgebra(Generic[BasisT]):
    """
    Internal base class for dense algebra elements with static basis metadata.

    This class is primarily intended to be subclassed to define concrete dense
    algebra types such as ``DenseLie``, ``DenseFreeTensor``, and
    ``DenseShuffleTensor``. It centralises storage, basic arithmetic, pytree
    registration, and simple basis-changing utilities shared by those classes.
    """

    data: jax.Array
    basis: BasisT

    DualVector: ClassVar[type["DenseAlgebra"]]

    def __init__(self, data: jax.typing.ArrayLike, basis: BasisT):
        self.basis = basis
        self.data = jnp.asarray(data)

        if not self.data.shape:
            raise ValueError("data must have at least one dimension")

        if not basis.size() == self.data.shape[-1]:
            raise ValueError(
                f"basis size must match data dimension, expected {basis.size()} but got {self.data.shape[-1]}"
            )

    @property
    def dtype(self):
        """Data type of the coefficient array."""
        return self.data.dtype

    @property
    def shape(self):
        """Full shape of the coefficient array."""
        return self.data.shape

    @property
    def batch_shape(self):
        """Leading batch dimensions of the coefficient array."""
        return self.data.shape[:-1]

    @property
    def dimension(self):
        """
        Dimension of the algebra

        Should match self.basis.size() in most cases.
        """
        return self.data.shape[-1]

    def change_depth(self: AlgebraT, new_depth: int) -> AlgebraT:
        """
        Re-express the element in the same basis family at a new depth.

        If ``new_depth`` matches the current depth, the element is returned
        unchanged. Otherwise, a new basis of the same concrete type and width is
        constructed and the coefficient data are resized accordingly.

        :param new_depth: Target truncation depth.
        :return: Algebra element represented in the new basis.
        """
        if new_depth == self.basis.depth:
            return self

        algebra_cls = type(self)
        basis_cls = type(self.basis)

        new_basis = basis_cls(self.basis.width, new_depth)
        new_size = new_basis.size()

        return algebra_cls(_redepth_data(self.data, new_size), new_basis)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.data, dtype=dtype, copy=copy)

    def __numpy_dtype__(self):
        return np.dtype(self.data.dtype)

    def __add__(self, other):
        if isinstance(other, type(self)):
            return _algebra_add(self, other, impl=jnp.add)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, type(self)):
            return _algebra_add(self, other, impl=jnp.subtract)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, DenseAlgebra):
            return NotImplemented
        return _algebra_scalar_multiply(self, jnp.asarray(other))

    def __rmul__(self, other):
        if isinstance(other, DenseAlgebra):
            return NotImplemented
        return _algebra_scalar_multiply(self, jnp.asarray(other))

    def __truediv__(self, other):
        if isinstance(other, DenseAlgebra):
            return NotImplemented
        return _algebra_scalar_multiply(self, 1 / jnp.asarray(other))

    def tree_flatten(self):
        return (self.data,), (self.basis,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.data = children[0]
        obj.basis = aux_data[0]
        return obj

    @classmethod
    def zero(
        cls: type[_T],
        basis: BasisT,
        dtype: jax.typing.DTypeLike = jnp.dtype("float32"),
        batch_dims: tuple[int, ...] = tuple(),
    ) -> _T:
        """
        Construct the additive identity in the given basis.

        The returned element has coefficient array shape
        ``batch_dims + (basis.size(),)`` and contains zeros in every basis
        coordinate. The optional batch dimensions allow a batch of zero
        elements to be created in a single call.

        :param basis: Basis in which the zero element should live.
        :param dtype: Data type used for the coefficient array.
        :param batch_dims: Optional leading batch dimensions.
        :return: A zero element of ``cls`` in ``basis``.
        """
        shape = batch_dims + (basis.size(),)
        zero_data = jnp.zeros(dtype=jnp.dtype(dtype), shape=shape)
        return cls(zero_data, basis)


DenseAlgebra.DualVector = DenseAlgebra


def to_dual(algebra: DenseAlgebra) -> DenseAlgebra:
    """
    Map an algebra element to its isomorphic dual-space representation.

    This reinterprets ``algebra`` in the corresponding dual space using the
    dual basis associated with the same truncated algebra. In the truncated
    algebras used in roughpy-jax (free, shuffle, and Lie), this map is
    represented by changing the concrete algebra type while leaving the
    coefficient data and basis metadata unchanged.

    :param algebra: The algebra element to reinterpret in the dual space.
    :return: The same coefficients viewed in the dual algebra type.
    """
    return algebra.DualVector(algebra.data, algebra.basis)


def add(lhs, rhs):
    """
    Add dense algebra operands, falling back to ``jnp.add`` otherwise.

    :param lhs: Left operand.
    :param rhs: Right operand.
    :return: Sum in dense-algebra or array form, depending on the inputs.
    """
    if isinstance(lhs, DenseAlgebra) or isinstance(rhs, DenseAlgebra):
        return lhs + rhs
    return jnp.add(lhs, rhs)


def subtract(lhs, rhs):
    """
    Subtract dense algebra operands, falling back to ``jnp.subtract`` otherwise.

    :param lhs: Left operand.
    :param rhs: Right operand.
    :return: Difference in dense-algebra or array form, depending on the inputs.
    """
    if isinstance(lhs, DenseAlgebra) or isinstance(rhs, DenseAlgebra):
        return lhs - rhs
    return jnp.subtract(lhs, rhs)


def multiply(lhs, rhs):
    """
    Multiply dense algebra operands, falling back to ``jnp.multiply`` otherwise.

    For dense algebra objects this dispatches to scalar multiplication.

    :param lhs: Left operand.
    :param rhs: Right operand.
    :return: Product in dense-algebra or array form, depending on the inputs.
    """
    if isinstance(lhs, DenseAlgebra) or isinstance(rhs, DenseAlgebra):
        return lhs * rhs
    return jnp.multiply(lhs, rhs)


def divide(lhs, rhs):
    """
    Divide dense algebra operands, falling back to ``jnp.divide`` otherwise.

    For dense algebra objects this dispatches to scalar division.

    :param lhs: Left operand.
    :param rhs: Right operand.
    :return: Quotient in dense-algebra or array form, depending on the inputs.
    """
    if isinstance(lhs, DenseAlgebra) or isinstance(rhs, DenseAlgebra):
        return lhs / rhs
    return jnp.divide(lhs, rhs)


def negative(arg):
    """
    Negate a dense algebra element or array-like argument.

    :param arg: Operand to negate.
    :return: Negated value with basis preserved for dense algebra inputs.
    """
    if isinstance(arg, DenseAlgebra):
        return type(arg)(jnp.negative(arg.data), arg.basis)
    return jnp.negative(arg)


def matmul(lhs, rhs):
    """
    Matrix-multiply operands, deferring to algebra ``@`` when available.

    :param lhs: Left operand.
    :param rhs: Right operand.
    :return: Matrix product in dense-algebra or array form, depending on the inputs.
    """
    if isinstance(lhs, DenseAlgebra) or isinstance(rhs, DenseAlgebra):
        return lhs @ rhs
    return jnp.matmul(lhs, rhs)
