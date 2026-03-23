import typing
from collections.abc import Hashable, Iterable
from typing import Literal, TypeVar

import jax
import numpy as np
from roughpy import compute as rpc


@typing.runtime_checkable
class Basis(typing.Protocol, Hashable):
    """
    Structural protocol shared by basis objects used in ``roughpy_jax``.

    Any object implementing this protocol provides the width, truncation depth,
    and degree offsets needed to construct compatible tensor or Lie bases.
    """

    width: np.int32
    depth: np.int32
    degree_begin: np.ndarray[np.int64.dtype]

    def size(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...


BasisT = TypeVar("BasisT", bound=Basis)


def _basis_tree_flatten(basis):
    return (), basis


def _basis_tree_unflatten(aux_data, _children):
    return aux_data


class TensorBasis(rpc.TensorBasis):
    """
    Word basis for the tensor algebra and shuffle algebra.

    Basis elements are indexed by words in the alphabet of ``width`` letters,
    truncated at words of length ``depth``.
    """


jax.tree_util.register_pytree_node(
    TensorBasis, _basis_tree_flatten, _basis_tree_unflatten
)


class LieBasis(rpc.LieBasis):
    """
    An instance of a Hall basis for the Lie algebra.

    A Hall basis is indexed by integer keys k > 0. To each key there is an
    associated pair of parents (a, b) where a and b are both keys belonging
    to the Hall basis. The exceptions are the "letters", which are those keys
    k for which the parents are (0, k). For convenience, we usually add a null
    element to the basis at key 0 and with parents (0, 0), which serves to
    offset elements correctly. However, this is not a valid key for the vectors
    and thus the key to index map subtracts 1 from the key to obtain the
    position in the vector.

    The default constructor requires only width and depth and constructs a
    Hall set greedily, minimizing the degree of the left parent. For instance,
    for width 2 and depth 4, the basis contains 5 keys 1 -> (0, 1), 2 -> (0, 2),
    3 -> (1, 2) (which represents the bracket [1,2]), 4 -> (1, 3) ([1,[1,2]]),
    and 5 -> (2, 3) ([2,[1,2]]).

    This implementation is designed to be flexible as to the exact contents of
    the Hall set, provided it is given in the format described above. The basis
    must also be ordered by degree, so elements of degree k must appear
    sequentially and between elements of degree k - 1 and degree k + 1 (if such
    elements exist).
    """


jax.tree_util.register_pytree_node(LieBasis, _basis_tree_flatten, _basis_tree_unflatten)


def check_basis_compat(
    first_basis: Basis,
    *other_bases: Basis,
    exact: bool = False,
    same_type: bool = False,
) -> None:
    """
    Check that a collection of basis objects are width-compatible.

    Basis compatibility in roughpy-jax is currently defined by equal width.

    :param same_type: Check if all the bases of the same type
    :param exact: Check if all bases have the same width and depth
    :param first_basis: Reference basis used for compatibility checks.
    :param other_bases: Additional basis objects to validate against ``first_basis``.
    :raises ValueError: If any basis has a different width to ``first_basis``.
    """
    for i, basis in enumerate(other_bases):
        if (
            basis.width != first_basis.width
            or (exact and basis.depth != first_basis.depth)
            or (same_type and type(basis) is not type(first_basis))
        ):
            raise ValueError(f"Incompatible width between basis 0 and basis {i + 1}")


def result_basis(
    *bases: BasisT, strategy: Literal["first", "max_depth", "min_depth"] = "max_depth"
) -> BasisT:
    """
    Select a result basis from a compatible collection of same-type bases.

    This helper first checks that all supplied bases are compatible and of the
    same concrete basis type, then selects one of them using a built-in
    strategy. The available strategies are ``"first"``, ``"max_depth"``, and
    ``"min_depth"``.

    The candidate bases may be supplied either as positional arguments or as a
    single iterable of bases.

    :param bases: Candidate bases to select from.
    :param strategy: Selection strategy used to choose the result basis.
    :return: The selected basis.
    :raises ValueError: If no bases are supplied, if the bases are incompatible,
        or if an unknown strategy is requested.
    """
    if (
        len(bases) == 1
        and isinstance(bases[0], Iterable)
        and not isinstance(bases[0], Basis)
    ):
        bases = tuple(bases[0])

    if not bases:
        raise ValueError("expected at least one basis")

    basis, *tail = bases
    check_basis_compat(basis, *tail, same_type=True)

    if strategy == "first":
        return basis

    if strategy == "max_depth":
        return max(bases, key=lambda item: item.depth)

    if strategy == "min_depth":
        return min(bases, key=lambda item: item.depth)

    raise ValueError(f"unknown basis selection strategy: {strategy}")


def to_tensor_basis(basis: Basis) -> TensorBasis:
    """
    Construct the tensor basis corresponding to ``basis``.

    This helper accepts any basis-like object and returns the word basis for
    the tensor algebra and shuffle algebra with matching width and depth. It is
    used to construct a consistent algebra environment.

    :param basis: Basis-like object providing ``width`` and ``depth``.
    :return: Tensor basis with the same width and depth as ``basis``.
    """
    return TensorBasis(basis.width, basis.depth)


def to_lie_basis(basis: Basis) -> LieBasis:
    """
    Construct the Lie basis corresponding to ``basis``.

    This helper accepts any basis-like object and returns the Lie basis with
    matching width and depth. It is used to construct a consistent
    a algebra environment.

    :param basis: Basis-like object providing ``width`` and ``depth``.
    :return: Lie basis with the same width and depth as ``basis``.
    """
    return LieBasis(basis.width, basis.depth)
