#  Copyright (c) 2023 the RoughPy Developers. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification,
#  are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#  may be used to endorse or promote products derived from this software without
#  specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
#  OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
#  USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import roughpy
from roughpy import FreeTensor, Monomial, ShuffleTensor


def test_construct_tensor_poly_coeffs():
    data = [1 * Monomial(f"x{i}") for i in range(3)]

    ft = FreeTensor(data, width=2, depth=2, dtype=roughpy.RationalPoly)

    assert str(ft) == "{ { 1(x0) }() { 1(x1) }(1) { 1(x2) }(2) }"


def test_exp_log_roundtrip_poly_coeffs():
    data = [0, 1 * Monomial('x1'), 1 * Monomial('x2')]
    ft = FreeTensor(data, width=2, depth=2, dtype=roughpy.RationalPoly)

    assert ft.exp().log() == ft


def test_shuffle_product_poly_coeffs():
    lhs = ShuffleTensor([1 * Monomial(f"x{i}") for i in range(7)], width=2,
                        depth=2, dtype=roughpy.RationalPoly)
    rhs = ShuffleTensor([1 * Monomial(f"y{i}") for i in range(7)], width=2,
                        depth=2, dtype=roughpy.RationalPoly)

    result = lhs * rhs

    expected_data = [
        1 * Monomial('x0') * Monomial('y0'),
        Monomial('x0') * Monomial('y1')
        + Monomial('x1') * Monomial('y0'),
        Monomial('x0') * Monomial('y2')
        + Monomial('x2') * Monomial('y0'),
        Monomial(['x0', 'y3'])
        + 2 * Monomial(['x1', 'y1'])
        + Monomial(['x3', 'y0']),
        Monomial(['x0', 'y4'])
        + Monomial(['x1', 'y2'])
        + Monomial(['y1', 'x2'])
        + Monomial(['x4', 'y0']),
        Monomial(['x0', 'y5'])
        + Monomial(['x2', 'y1'])
        + Monomial(['y2', 'x1'])
        + Monomial(['x5', 'y0']),
        Monomial(['x0', 'y6'])
        + 2 * Monomial(['x2', 'y2'])
        + Monomial(['x6', 'y0'])
    ]
    expected = ShuffleTensor(expected_data, width=2, depth=2,
                             dtype=roughpy.RationalPoly)

    assert result == expected, f"{expected - result} != 0"
