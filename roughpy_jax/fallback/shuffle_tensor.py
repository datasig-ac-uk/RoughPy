from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from roughpy_jax import TensorBasis
from roughpy_jax.fallback.modifiers import _identity_modifier



@jax.vmap
@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8))
def st_fma_fallback(a_data: jnp.ndarray,
                    b_data: jnp.ndarray,
                    c_data: jnp.ndarray,
                    basis: TensorBasis,
                    b_max_degree: np.int32,
                    c_max_degree: np.int32,
                    modifier: Callable[[jnp.ndarray], jnp.ndarray] = _identity_modifier,
                    b_min_degree: np.int32 = 0,
                    c_min_degree: np.int32 = 0
                    ) -> jnp.ndarray:
    """
    Fallback implementation for the shuffle tensor fused multiply-add to use when no
    accelerated implementation is available.

    This is equivalent to the fused multiply-add d = a + b * c, where here * denotes
    the shuffle product of (co)tensors when the modifier is the identity. When a is
    not used in another computation, the jit compiler might perform this operation
    in-place.

    This function should not be used directly, and should only be used via the st_fma
    function.

    :param a_data: Data for (co)tensor a
    :param b_data: Data for (co)tensor b
    :param c_data: Data for (co)tensor c
    :param basis: basis for the result
    :param b_max_degree: Maximum degree for the b (co)tensor
    :param c_max_degree: Maximum degree for the c (co)tensor
    :param modifier: Modifier function to be applied to the shuffle product b*c (default identity)
    :param b_min_degree: Optional min degree for the b (co)tensor (default 0)
    :param c_min_degree: Optional min degree for the c (co)tensor (default 0)
    :return: Array contents of the result d of a + b*c
    """

    db = basis.degree_begin
    width = basis.width

    out_max_degree = min(np.int32(basis.max_degree), b_max_degree + c_max_degree)
    out_min_degree = b_min_degree + c_min_degree

    # We have to unpack a word into individual letters so they can be repacked
    # according to the shuffle mask. This is a repeatedly taking off the last
    # remainder after division by width and sticking it in a list (in left-to-
    # right order) These are two helper functions that decompose then recompose
    # the one index into a tuple and back into a pair of indexes (and respective
    # degrees).
    def idx_to_word(degree, idx):
        word = [-1 for _ in range(degree)]

        for i in range(degree):
            word[i] = idx % width
            idx /= width

        return tuple(word)

    def repack_words(mask, word):
        left_idx = 0
        left_deg = 0
        right_idx = 0
        right_deg = 0

        for i in range(len(word)):
            if mask & 1 == 1:
                left_idx *= width
                left_idx += word[i]
                left_deg += 1
            else:
                right_idx *= width
                right_idx += word[i]
                right_deg += 1

            mask >>= 1

        return (left_idx, left_deg), (right_idx, right_deg)

    # The inner-most loop is over the shuffle mask, which ranges from 0 (all letters in left)
    # to 2^degree - 1 (all letters in right). We skip any masks where the left or right degree
    # is out of bounds otherwise accumulate the sum of word pairs resulting from the mask
    # and outer word, and then add this into the output array (applying modifier as appropriate)
    def shuffle_idx(i, val, *, degree):
        word = idx_to_word(degree, i)

        def inner(mask, acc):
            (left_idx, left_deg), (right_idx, right_deg) = repack_words(mask, word)

            if left_deg < b_min_degree or left_deg > b_max_degree:
                return acc

            if right_deg < c_min_degree or right_deg > c_max_degree:
                return acc

            return acc + b_data[db[left_deg]+left_idx] * c_data[db[right_deg]+right_idx]

        x = jax.lax.fori_loop(0, 1 << degree, inner, 0, unroll=True)

        return val.at[db[degree] + i].add(modifier(x))

    # Loop this over all the indices in the degree
    def shuffle_layer(d, val):
        return jax.lax.fori_loop(0, db[d + 1] - db[d], partial(shuffle_idx, degree=d), val, unroll=True)

    # Handle the very simple unit element separately
    if out_min_degree == 0:
        a_data = a_data.at[0].add(modifier(b_data[0] * c_data[0]))

    # Outer loop over all the non-zero degrees that we care about.
    return jax.lax.fori_loop(max(np.int32(1), out_min_degree), out_max_degree, shuffle_layer, a_data)