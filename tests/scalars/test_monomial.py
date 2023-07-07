
import pytest

import roughpy
from roughpy import Monomial







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
