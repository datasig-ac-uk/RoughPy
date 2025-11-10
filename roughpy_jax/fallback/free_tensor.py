
from functools import partial
from typing import Callable, Any

import jax
import jax.numpy as jnp
import numpy as np

from roughpy_jax import TensorBasis
from roughpy_jax.fallback.modifiers import _identity_modifier



# TODO: This needs an appropriate vmap decorator
@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8))
def ft_fma_fallback(a_data: jnp.ndarray,
                     b_data: jnp.ndarray,
                     c_data: jnp.ndarray,
                     basis: TensorBasis,
                     b_max_deg: np.int32,
                     c_max_deg: np.int32,
                     modifier: Callable[[jnp.ndarray], jnp.ndarray] = _identity_modifier,
                     b_min_deg: np.int32 = 0,
                     c_min_deg: np.int32 = 0,
                     ) -> jnp.ndarray:
    """
    Fallback implementation to use when no accelerated implementation is available for
    the free tensor fused multiply-add operation. Returns the data for the result.

    This is equivalent to the fused multiply-add operation d = a + b * c when the
    modifier is the identity modifier. Here * denotes free tensor multiplication
    When a is not used again in a computation, the jit compiler might perform this inplace.

    This function should not be used directly and instead should be called via the ft_fma
    function.

    :param a_data: Data for tensor a
    :param b_data: Data for tensor b
    :param c_data: Data for tensor c
    :param basis: The basis for the result tensor
    :param b_max_deg: Maximum degree for the b tensor
    :param c_max_deg: Maximum degree for the c tensor
    :param modifier: Modifier function to be applied to the product b * c, (default identity)
    :param b_min_deg: Optional min degree for terms from b (default 0)
    :param c_min_deg: Optional min degree for terms from c (default 0)
    :return: Array contents of the result d of a + b*c (where * is free-tensor multiplication)
    """
    db = basis.degree_begin

    a_max_deg = min(np.int32(basis.depth), b_max_deg + c_max_deg)
    a_min_deg = b_min_deg + c_min_deg

    def deg_d_update(bdeg, val, *, a_deg):
        cdeg = a_deg - bdeg
        b_level = b_data[db[bdeg]:db[bdeg+1]]
        c_level = c_data[db[cdeg]:db[cdeg+1]]
        return val.at[db[a_deg]:db[a_deg+1]].add(modifier(jnp.outer(b_level, c_level).ravel()))

    def level_d_func(d, val):
        a_deg = a_max_deg - d
        level_f = partial(deg_d_update, a_deg=a_deg)

        left_deg_start = max(b_min_deg, a_deg - c_max_deg)
        left_deg_end = min(b_max_deg, a_deg - c_min_deg)

        return jax.lax.fori_loop(left_deg_start, left_deg_end + 1, level_f, val, unroll=True)

    return jax.lax.fori_loop(a_min_deg, a_max_deg, level_d_func, a_data)


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7))
def ft_inplace_mul_fallback(a_data: jnp.ndarray,
                            b_data: jnp.ndarray,
                            basis: TensorBasis,
                            a_max_deg: np.int32,
                            b_max_deg: np.int32,
                            modifier: Callable[[jnp.ndarray], jnp.ndarray] = _identity_modifier,
                            a_min_deg: np.int32 = 0,
                            b_min_deg: np.int32 = 0):
    """
    Fallback implementation to use when no accelerated implementation is available for
    the inplace multiplication of free tensors a *= b (where * denotes free tensor
    multiplication).

    This performs a (potentially modified) free tensor multiplication, which might
    be jit compiled into an inplace operation. This is much trickier to get right
    because operations have to be done in a certain order to make sure the parts
    of the a_data array are only updated once they are no longer needed for other
    parts of the computation. JAX may or may not be able to reason that it is safe
    to replace these with an in-place operation, although it is (theoretically)
    guaranteed to be so.

    This function should not be used directly. It is mostly for implementing other
    fallback operations.

    :param a_data: Data for left-hand/result tensor
    :param b_data: Data for right-hand tensor
    :param basis: Basis for left-hand/result tensor
    :param a_max_deg: Maximum degree for the left-hand tensor
    :param b_max_deg: Maximum degree for the right-hand tensor
    :param modifier: Modifier to apply to the product
    :param a_min_deg: Optional minimum degree for the left-hand tensor
    :param b_min_deg: Optional minimum degree for the right-hand tensor

    :return: The data for the result of a * b (or a *= b)
    """
    db = basis.degree_begin

    # This is pretty much as before, except val is both the output and the lhs input
    def deg_d_update(bdeg, val, *, out_deg):
        cdeg = out_deg - bdeg
        a_level = val[db[bdeg]:db[bdeg+1]]
        b_level = b_data[db[cdeg]:db[cdeg+1]]
        return val.at[db[out_deg]:db[out_deg+1]].add(modifier(jnp.outer(a_level, b_level).ravel()))

    def level_d_func(d, val):
        ## Here it's very important that we start from the top
        out_deg = a_max_deg - d

        # This level function cannot be used to update the a_deg = out_deg case
        level_f = partial(deg_d_update, out_deg=out_deg)

        # Update the top level, the if statement is statically determined
        if b_min_deg > 0:
            val = val.at[db[out_deg]:db[out_deg+1]].set(0)
        else:
            top_level = val.at[db[out_deg]:db[out_deg+1]]
            val = val.at[db[out_deg]:db[out_deg+1]].set(modifier(top_level * b_data[0]))


        left_deg_start = max(b_min_deg, out_deg - b_max_deg)
        left_deg_end = min(b_max_deg, out_deg - b_min_deg)

        # The fori_loop finishes before the left_deg_end rather than at left_deg_end as
        # in the ft_fma implementation
        return jax.lax.fori_loop(left_deg_start, left_deg_end, level_f, val, unroll=True)

    return jax.lax.fori_loop(a_min_deg, a_max_deg, level_d_func, a_data)




@partial(jax.jit, static_argnums=(2, 3))
def antipode_fallback(out_data: jnp.ndarray, arg_data: jnp.ndarray, basis: TensorBasis, no_sign: bool) -> jnp.ndarray:
    """
    Fallback implementation of the tensor antipode to be used when no accelerated method is
    available.

    This operation performs the antipode, which is essentially a level-wise transposition
    combined with multiplication by (-1)^degree.

    This should not be used directly.

    :param out_data: data for the output tensor
    :param arg_data: data for the input tensor
    :param basis: Basis for both tensors
    :param no_sign: Flag to indicate if we should not sign tensors
    :return: the out_data array filled with the antipode of arg_data.
    """
    db = basis.degree_begin
    width = basis.width

    def transpose_level(i, val):
        sign = 1 if (no_sign or i % 2 == 0) else -1
        level_data = arg_data[db[i]:db[i+1]].reshape((width,) * i)
        return val.at[db[i]:db[i+1]].set(sign*jnp.transpose(level_data).ravel())

    return jax.lax.fori_loop(0, basis.depth, transpose_level, out_data, unroll=True)
