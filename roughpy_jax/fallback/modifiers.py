
import jax
import jax.numpy as jnp




# These modifiers are useful for invoking specific behaviours in the fma routines.


@jax.jit
def _identity_modifier(a: jnp.ndarray) -> jnp.ndarray:
    """
    Identity function, returns argument as is

    :param a: argument array
    :return: a unchanged
    """
    return a


@jax.jit
def _negate_modifier(a: jnp.ndarray) -> jnp.ndarray:
    """
    Negation function, returns argument but with opposite sign

    :param a:  argument array
    :return:  negative of a
    """
    return -a


def _make_divide_by_int_modifier(val: int) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Create a new modifier that divides the entries by an integer value

    :param val: Integer to divide by
    :return: Modifier function that divides the elements in the array by the given integer
    """
    def inner(a: jnp.ndarray) -> jnp.ndarray:
        return a / val

    return jax.jit(inner)
