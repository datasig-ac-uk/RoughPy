from typing import Any, Callable, Iterable
import numpy as np
from numpy.testing import assert_allclose
import roughpy_jax as rpj


def _is_tensor(x: Any) -> bool:
    return isinstance(x, (rpj.FreeTensor, rpj.ShuffleTensor, rpj.Lie))


def _data(x: Any) -> Any:
    return x.data if _is_tensor(x) else x


def _binary_tensor_op(x: Any, y: Any, fn: Callable[[Any, Any], Any]) -> Any:
    """
    Binary operation between x and y, unpacking as tensor if appropriate.

    Relaxed constraints if other side not a tensor, broadcast as appropriate.
    """
    if _is_tensor(x) or _is_tensor(y):
        z_data = fn(_data(x), _data(y))
        z_like = x if _is_tensor(x) else y
        z_cls = type(z_like)
        return z_cls(z_data, z_like.basis)
    else:
        return fn(x, y)


def _add(x: Any, y: Any) -> Any:
    return _binary_tensor_op(x, y, lambda a, b: a + b)


def _mul(x: Any, y: Any) -> Any:
    return _binary_tensor_op(x, y, lambda a, b: a * b)


def _sub(x: Any, y: Any) -> Any:
    return _binary_tensor_op(x, y, lambda a, b: a - b)


def _div(x: Any, y: Any) -> Any:
    return _binary_tensor_op(x, y, lambda a, b: a / b)


def assert_is_linear(
    fn: Callable[..., Any],
    x: Any,
    y: Any,
    alpha: Any,
    beta: Any,
    abs_tol: float = 1e-6,
    rel_tol: float = 1e-6,
):
    """
    Validates whether a given function adheres to the principle of linearity by checking if the
    output of the function is consistent with the linear combination of its inputs. Linearity is
    tested by verifying if `fn(alpha * x + beta * y)` is approximately equal to
    `alpha * fn(x) + beta * fn(y)` within the given absolute and relative tolerances.

    :param fn: The function to validate. It should accept and return values of types compatible
               with the linearity assumptions being tested.
    :param x: First input to the function `fn` involved in the linearity check.
    :param y: Second input to the function `fn` involved in the linearity check.
    :param alpha: Scalar multiplier for the first input `x` in the linear combination.
    :param beta: Scalar multiplier for the second input `y` in the linear combination.
    :param abs_tol: Absolute tolerance for the comparison of function outputs to determine
                    linearity. Default is 1e-6.
    :param rel_tol: Relative tolerance for the comparison of function outputs to determine
                    linearity. Default is 1e-6.
    :return: None. The function asserts and raises an error if the function `fn` is found
             to not be linear within the specified tolerances.
    """
    z = _add(_mul(alpha, x), _mul(beta, y))
    fn_inner = fn(z)
    fn_outer = _add(_mul(alpha, fn(x)), _mul(beta, fn(y)))

    lhs = _data(fn_inner)
    rhs = _data(fn_outer)

    assert_allclose(lhs, rhs, atol=abs_tol, rtol=rel_tol)


def assert_is_adjoint(
    fn: Callable[..., Any],
    fn_adj: Callable[..., Any],
    x: Any,
    functional: Any,
    domain_pairing: Callable[[Any, Any], Any],
    codomain_pairing: Callable[[Any, Any], Any],
    abs_tol: float = 1e-6,
    rel_tol: float = 1e-6,
):
    """
    Checks whether the provided function and its adjoint satisfy the adjoint property within
    specified tolerances. This function compares the domain pairing of the adjoint applied
    to the functional and the variable with the codomain pairing of the functional and the
    original function applied to the variable.

    :param fn: The primary function being evaluated against its adjoint.
    :param fn_adj: The adjoint of the primary function to be verified.
    :param x: The variable input to the function and adjoint for evaluation.
    :param functional: A functional being applied to test the adjoint property.
    :param domain_pairing: A callable that establishes the pairing in the domain space
        between the adjoint function applied to the functional and the variable.
    :param codomain_pairing: A callable that establishes the pairing in the codomain
        space between the functional and the primary function applied to the variable.
    :param abs_tol: Absolute tolerance for the comparison. Defaults to 1e-6.
    :param rel_tol: Relative tolerance for the comparison. Defaults to 1e-6.
    :return: None. Ensures the adjoint property is satisfied based on input and tolerances,
        raising an assertion error if not.
    """
    lhs = _data(domain_pairing(fn_adj(functional), x))
    rhs = _data(codomain_pairing(functional, fn(x)))

    assert_allclose(lhs, rhs, atol=abs_tol, rtol=rel_tol)


def assert_is_derivative(
    fn: Callable[..., Any],
    fn_deriv: Callable[..., Any],
    x: Any,
    tangent: Any,
    eps_factors: Iterable[float] = (1.0e-3, 1.0e-6, 1.0e-9),
    abs_tol: float = 1e-6,
    rel_tol: float = 1e-6,
):
    """
    Asserts that a given function's derivative is accurate within specified tolerances
    by comparing it to finite difference approximations of the derivative.

    The method verifies if a user-supplied derivative function produces results consistent
    with the derivative of the function obtained via finite difference approximations.
    It does this by perturbing the input along a specified tangent direction using various
    scaling factors (`eps_factors`) and comparing the analytical derivative to the
    finite difference derivative within absolute (`abs_tol`) and relative (`rel_tol`) tolerances.

    :param fn: Callable function representing the primary function to differentiate.
    :param fn_deriv: Callable function representing the derivative of the primary function.
    :param x: The point at which the derivative of the function is evaluated.
    :param tangent: A tangent direction vector along which finite differences will be computed.
    :param eps_factors: Iterable of scaling factors used to perturb the input for finite
        difference computation of the derivative.
    :param abs_tol: Absolute tolerance for verifying the computed derivative's accuracy.
    :param rel_tol: Relative tolerance for verifying the computed derivative's accuracy.
    :return: None. The function performs assertions but does not return any value.
    """
    fx = fn(x)

    for eps in eps_factors:
        approx = _div(_sub(fn(_add(x, _mul(eps, tangent))), fx), eps)

        lhs = _data(approx)
        rhs = _data(fn_deriv(x, tangent))
        atol = abs_tol + eps
        rtol = rel_tol + eps

        assert_allclose(lhs, rhs, atol=atol, rtol=rtol)


def assert_is_adjoint_derivative(
    fn: Callable[..., Any],
    fn_adj_deriv: Callable[..., Any],
    x: Any,
    tangent: Any,
    cotangent: Any,
    domain_pairing: Callable[[Any, Any], Any],
    codomain_pairing: Callable[[Any, Any], Any],
    eps_factors: Iterable[float] = (1.0e-3, 1.0e-6, 1.0e-9),
    abs_tol: float = 1e-6,
    rel_tol: float = 1e-6,
):
    """
    Asserts that the provided adjoint derivative function is consistent with the function's
    tangent linear behavior. This function evaluates the relationship between the function,
    its adjoint derivative, a tangent, and a cotangent using finite difference approximations.

    :param fn: The function for which the adjoint derivative consistency is being tested.
        It should take an input of type compatible with `x` and return a result of
        appropriate type that the pairing and other operations can handle.
    :param fn_adj_deriv: The adjoint derivative of the input function `fn`. It should
        map the input `x` and cotangent to the tangent space, effectively representing
        the dual map.
    :param x: The point in the domain of `fn` at which the adjoint derivative is verified.
    :param tangent: An element in the tangent space at `x`, used to evaluate the property
        of the adjoint derivative in comparison to finite differences.
    :param cotangent: An element in the cotangent space at the output of `fn`, used to
        test the duality between the adjoint derivative and finite difference approximations.
    :param domain_pairing: A callable that pairs tangent and cotangent information in
        the domain space.
    :param codomain_pairing: A callable that pairs cotangent and chord information in
        the codomain space.
    :param eps_factors: An iterable of floats representing the step sizes used for
        the finite difference approximation. The smaller the step size, the closer the
        approximation to the true derivative, subject to numerical precision limitations.
    :param abs_tol: Absolute tolerance for the equality assertion between the finite
        difference approximation and the adjoint derivative's evaluation.
    :param rel_tol: Relative tolerance for the equality assertion between the finite
        difference approximation and the adjoint derivative's evaluation.
    :return: None. Raises an assertion error if the adjoint derivative does not align
        with the finite difference approximations with specified tolerances.
    """
    fx = fn(x)

    for eps in eps_factors:
        chord_eval = codomain_pairing(cotangent, _div(fn(_sub(_add(x, _mul(eps, tangent))), fx), eps))
        adjoint_eval = domain_pairing(fn_adj_deriv(x, cotangent), tangent)

        lhs = _data(chord_eval)
        rhs = _data(adjoint_eval)
        atol = abs_tol + eps
        rtol = rel_tol + eps

        assert_allclose(lhs, rhs, atol=atol, rtol=rtol)
