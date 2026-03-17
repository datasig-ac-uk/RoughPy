from typing import TypeVar, Callable, Generic, ClassVar, Type

import jax
import jax.numpy as jnp
import numpy as np

# from roughpy_jax.bases import BasisT
BasisT = TypeVar("BasisT")  ## TODO: replace when basis utilities is merged

AlgebraT = TypeVar("AlgebraT")
_T = TypeVar("_T")


def get_batch_shape(operand) -> tuple[int, ...]:
    """
    Return the batch shape associated with an operand.

    Dense algebra objects contribute their ``batch_shape`` property, i.e. the
    leading dimensions before the trailing algebra coordinate. Plain arrays are
    treated as pure batch data and therefore contribute their full shape.

    :param operand: Dense algebra object or array-like operand.
    :return: Batch shape for the operand.
    """
    if (batch_shape := getattr(operand, "batch_shape", None)) is not None:
        return batch_shape

    return jnp.shape(operand)


def get_common_batch_shape(*operands) -> tuple[int, ...]:
    """
    Validate that all operands share the same batch shape and return it.

    Dense algebra objects contribute the leading dimensions before the trailing
    algebra coordinate. Plain arrays contribute their full shape. At present,
    operations in roughpy-jax require all operands to have identical batch
    shape, and this helper returns that common value after validation.

    :param operands: Dense algebra objects or array-like operands to validate.
    :return: The batch shape common to all operands.
    :raises ValueError: If no operands are supplied or any operand has a
        different batch shape.
    """
    if not operands:
        raise ValueError("expected at least one operand")

    first, *rem = operands
    batch_shape = get_batch_shape(first)

    for i, operand in enumerate(rem, start=1):
        operand_batch_shape = get_batch_shape(operand)
        if operand_batch_shape != batch_shape:
            raise ValueError(
                f"incompatible batch shape in argument at index {i}:"
                f" expected {batch_shape} but got {operand_batch_shape}"
            )

    return batch_shape


def broadcast_to_batch_shape(
    data: jax.typing.ArrayLike,
    batch_shape: tuple[int, ...],
    *,
    core_dims: int = 1,
) -> jax.Array:
    """
    Reshape data for broadcasting over a target batch shape and core dimensions.

    Scalars and batch prefixes are reshaped by appending singleton dimensions
    until they are broadcast-compatible with arrays of shape
    ``batch_shape + core_shape``. Inputs already shaped as
    ``batch_shape + (1,) * core_dims`` are returned unchanged.

    :param data: Array-like input to reshape.
    :param batch_shape: Target batch shape.
    :param core_dims: Number of trailing core dimensions.
    :return: Reshaped JAX array.
    :raises ValueError: If ``data`` is incompatible with ``batch_shape``.
    """
    data = jnp.asarray(data)
    data_shape = data.shape

    ds_len = len(data_shape)
    bs_len = len(batch_shape)

    if ds_len <= bs_len and data_shape == batch_shape[:ds_len]:
        new_shape = data_shape + (1,) * (bs_len - ds_len) + (1,) * core_dims
    elif ds_len == bs_len + core_dims and data_shape[:-core_dims] == batch_shape:
        if data_shape[-core_dims:] != (1,) * core_dims:
            raise ValueError(
                f"batch shape {batch_shape} is incompatible with data shape {data_shape}"
            )
        new_shape = data_shape
    else:
        raise ValueError(
            f"batch shape {batch_shape} is incompatible with data shape {data_shape}"
        )

    return jnp.reshape(data, new_shape)


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
    scalar = jnp.asarray(s)
    ext_scalar = broadcast_to_batch_shape(scalar, a.batch_shape)
    result_data = jnp.multiply(a.data, ext_scalar)
    return cls(result_data, a.basis)


def _redepth_data(data: jax.Array, new_alg_dim: int) -> jax.Array:
    """
    Resize the trailing algebra dimension by truncating or zero-padding.

    This helper is used when changing the truncation depth of a dense algebra
    element. Shrinking the algebra dimension drops higher-order coordinates,
    while increasing it appends zeros for the newly introduced basis elements.

    :param data: Coefficient array to resize.
    :param new_alg_dim: Target size of the trailing algebra dimension.
    :return: Resized coefficient array.
    """
    shape = data.shape

    if new_alg_dim < shape[-1]:
        return data[..., :new_alg_dim]

    pad_dims = [(0, 0)] * (len(shape) - 1) + [(0, new_alg_dim - shape[-1])]
    return jnp.pad(data, pad_dims)


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


class DenseTensor(DenseAlgebra["TensorBasis"]):
    """
    Dense tensor algebra element.

    This class represents elements of the tensor algebra over a given basis.
    It is a subclass of ``DenseAlgebra`` and inherits its algebraic structure.
    """

    @classmethod
    def identity(
        cls: Type[_T],
        basis: "TensorBasis",
        dtype: jax.typing.DTypeLike = jnp.dtype("float32"),
        batch_dims: tuple[int, ...] = tuple(),
    ) -> _T:
        """
            Construct the multiplicative identity in a tensor basis.

        return cls(result_data, basis)
            The returned element has coefficient array shape
            ``batch_dims + (basis.size(),)``. Its unit coordinate is set to ``1``
            and all other coefficients are zero. This is valid for the tensor
            algebra subclasses whose product has the empty word as identity.

            :param basis: Tensor basis in which the identity should live.
            :param dtype: Data type used for the coefficient array.
            :param batch_dims: Optional leading batch dimensions.
            :return: The identity element of ``cls`` in ``basis``.
        """
        shape = batch_dims + (basis.size(),)
        data = jnp.zeros(dtype=jnp.dtype(dtype), shape=shape)
        data = data.at[..., 0].set(1)
        return cls(data, basis)


def identity_like(tensor, dtype=None):
def zero_like(algebra: _T, dtype: jax.typing.DTypeLike | None = None) -> _T:
    """
    Construct a zero element with the same type, shape, and basis as ``algebra``.

    The returned object keeps the basis metadata and concrete dense algebra
    class of the input while replacing all coefficients with zeros. An
    optional ``dtype`` may be supplied to override the data type of the
    resulting coefficient array.

    :param algebra: Dense algebra element whose structure should be copied.
    :param dtype: Optional data type for the zero coefficient array.
    :return: Zero element matching the input algebra structure.
    """
    data = jnp.zeros_like(algebra.data, dtype=dtype)
    return type(algebra)(data, algebra.basis)


def identity_like(tensor: _T, dtype: jax.typing.DTypeLike | None = None) -> _T:
    """
    Creates an identity-like object based on the structure and type of the provided tensor. The resulting object retains
    the basis of the input tensor but modifies its data to follow an identity pattern. The primary element indicating
    identity is set to 1, while the remaining structure of the data is zeroed out.

    :param tensor: Input tensor that provides the basis and structure for the resulting identity-like object.
    :type tensor: type of the input tensor
    :param dtype: Optional data type to apply to the resulting identity-like object's data. If not provided, the data type
        of the input tensor is used.
    :type dtype: Optional
    :return: A new object of the same type as the input tensor, with modified data representing an identity-like structure.
    :rtype: type of the input tensor
    """
    data = jnp.zeros_like(tensor.data, dtype=dtype)
    data = data.at[..., 0].set(1)
    return type(tensor)(data, tensor.basis)


def to_dual(algebra: DenseAlgebra) -> DenseAlgebra:
    """
    Map an algebra element to its isomorphic dual-space representation.

    This reinterprets ``algebra`` in the corresponding dual space using the
    dual basis associated with the same truncated algebra. In the truncated
    algebras used in roughpy-jax (free, shuffle, and Lie), this map is
    represented by changing the concrete algebra type while leaving the
    coefficient data and basis metadata unchanged.

    This operation is only valid for algebras whose dual is isomorphic and is given
    the dual basis. Care should be used when using this function to check
    that these conditions hold.

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
