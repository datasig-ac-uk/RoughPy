from fractions import Fraction

import pytest



import roughpy
from roughpy import Monomial

@pytest.fixture(params=[3, 3.1415, Fraction(22, 7)])
def scalar_val(request):
    return request.param

def test_construct_single_str():
    m = Monomial("x1")

    assert str(m) == 'x1'


def test_mul_monomials():
    m1 = Monomial('x1')
    m2 = Monomial('x2')

    assert str(m1*m2) == 'x1 x2'


def test_add_monomials_gives_polynomial():
    m1 = Monomial('x1')
    m2 = Monomial('x2')

    p = m1 + m2

    assert type(p) == roughpy.PolynomialScalar


def test_sub_monomials_gives_polynomial():
    m1 = Monomial('x1')
    m2 = Monomial('x2')

    p = m1 - m2

    assert type(p) == roughpy.PolynomialScalar


def test_scalar_plus_monomial(scalar_val):
    m = Monomial('x')

    p = scalar_val + m

    assert type(p) == roughpy.PolynomialScalar


def test_monomial_plus_scalar(scalar_val):
    m = Monomial('x')
    p = m + scalar_val

    assert type(p) == roughpy.PolynomialScalar


def test_scalar_mul_monomial(scalar_val):
    m = Monomial('x')

    p = scalar_val * m

    assert type(p) == roughpy.PolynomialScalar


def test_monomial_mul_scalar(scalar_val):
    m = Monomial('x')
    p = m * scalar_val

    assert type(p) == roughpy.PolynomialScalar


def test_monomial_div_scalar(scalar_val):
    m = Monomial('x')
    p = m / scalar_val

    assert type(p) == roughpy.PolynomialScalar
