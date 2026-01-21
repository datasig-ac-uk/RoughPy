import functools

import roughpy.compute as rpc

from roughpy.compute import (
    DenseFreeTensor as FreeTensor,
    DenseShuffleTensor as ShuffleTensor,
    DenseLie as Lie,
    TensorBasis,
    LieBasis,
)

from .context import AlgebraContext


def free_multiply(lhs, rhs) -> FreeTensor:
    """
    Multiplies two free tensors.

    Performs the multiplication operation between `lhs` and `rhs`, both of which are
    instances of `FreeTensor`. The result is another free tensor combining the
    structure of the operands.

    :param lhs: The left-hand side free tensor operand.
    :type lhs: FreeTensor
    :param rhs: The right-hand side free tensor operand.
    :type rhs: FreeTensor
    :return: A free tensor resulting from the multiplication of `lhs` and `rhs`.
    :rtype: FreeTensor
    """
    return rpc.ft_mul(lhs, rhs)


def shuffle_multiply(lhs, rhs) -> ShuffleTensor:
    """
    Shuffles the elements of the given input tensors `lhs` and `rhs` and performs
    a multiplication operation between them, returning the result in the form of
    a `ShuffleTensor`.

    :param lhs: The left-hand side tensor to be shuffled and multiplied.
    :type lhs: Tensor
    :param rhs: The right-hand side tensor to be shuffled and multiplied.
    :type rhs: Tensor
    :return: A `ShuffleTensor` object resulting from the shuffle and multiplication
        of the input tensors.
    :rtype: ShuffleTensor
    """
    return rpc.st_mul(lhs, rhs)


def half_shuffle_multiply(lhs, rhs) -> ShuffleTensor:
    raise NotImplementedError


def adjoint_to_free_multiply(op, arg) -> ShuffleTensor:
    """
    Compute the adjoint to free multiplication of the given operator and argument.

    This function performs the computation of the left multiplication of a tensor
    with the adjoint of a specified operator, resulting in a new tensor. This is
    useful in certain mathematical and computational frameworks involving tensor
    algebra and distributed computation.

    :param op: The operator to be adjoint and multiplied. This input should
               be compatible with the requirements of the operation.
    :type op: Any
    :param arg: The tensor to be left-multiplied by the adjoint of the operator.
                This input should match the expected structure and dimensions
                required by the operation.
    :type arg: Any
    :return: A tensor resulting from the adjoint-left multiplication of the given
             operator and argument.
    :rtype: ShuffleTensor
    """
    return rpc.ft_adjoint_left_mul(op, arg)


def cbh(*args: Lie, basis: LieBasis = None) -> Lie:
    """
    Calculate the Campbell-Baker-Hausdorff (CBH) combination for a given sequence of Lie algebra
    elements. If no elements are provided, it will return the zero Lie element for the specified basis.
    The method leverages tensor arithmetic to compute the CBH formula iteratively.

    :param args: A variable-length list of Lie algebra elements of type Lie.
    :param basis: An optional LieBasis instance representing the structure on which the algebra
        operates. If not provided, the basis of the first Lie element will be used.
    :return: A resulting Lie element obtained by applying the CBH formula on the input elements.
    :rtype: Lie
    :raises ValueError: If no arguments and no valid basis are provided.
    """
    if not args and not isinstance(basis, LieBasis):
        raise ValueError("Must provide at least one argument or a basis")

    if not args:
        return Lie.zero(basis)

    lbasis = basis or args[0].basis
    tbasis = TensorBasis(lbasis.width, lbasis.depth)

    if len(args) == 1:
        return args[0]

    identity = FreeTensor.identity(tbasis)
    tresult = functools.reduce(
        lambda acc, x: rpc.ft_fmexp(acc, rpc.lie_to_tensor(x, tbasis)), args, identity
    )
    return rpc.tensor_to_lie(rpc.ft_log(tresult), lbasis)
