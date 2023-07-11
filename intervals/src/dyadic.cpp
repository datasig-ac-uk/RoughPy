// Copyright (c) 2023 RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

//
// Created by user on 02/03/23.
//

#include <roughpy/intervals/dyadic.h>

#include <ostream>

using namespace rpy;
using namespace rpy::intervals;

Dyadic::operator param_t() const noexcept
{
    return ldexp(param_t(m_multiplier), -m_power);
}
Dyadic& Dyadic::move_forward(Dyadic::multiplier_t arg)
{
    m_multiplier += arg;
    return *this;
}
Dyadic& Dyadic::operator++()
{
    ++m_multiplier;
    return *this;
}
const Dyadic Dyadic::operator++(int)
{
    Dyadic result(*this);
    ++m_multiplier;
    return result;
}
Dyadic& Dyadic::operator--()
{
    --m_multiplier;
    return *this;
}
const Dyadic Dyadic::operator--(int)
{
    Dyadic result(*this);
    --m_multiplier;
    return result;
}
bool Dyadic::rebase(Dyadic::power_t resolution)
{
    if (m_multiplier == 0) {
        m_power = resolution;
        return true;
    } else if (resolution >= m_power) {
        m_multiplier = shift(m_multiplier, resolution - m_power);
        m_power = resolution;
        return true;
    }

    if (m_power >= std::numeric_limits<multiplier_t>::digits + resolution) {
        resolution = (m_power - std::numeric_limits<power_t>::digits) + 1;
    }

    power_t rel_resolution{m_power - resolution};

    // starting at relative resolution find the first n in decreasing order so
    // that 2^n divides k 2^0 always divides k so the action stops
    power_t r = rel_resolution;
    for (; (m_multiplier % int_two_to_int_power(r)) != 0; --r) {}

    power_t offset = r;
    m_multiplier /= int_two_to_int_power(offset);
    m_power -= offset;
    // pr();
    return resolution == m_power;
}

bool rpy::intervals::operator<(const Dyadic& lhs, const Dyadic& rhs)
{
    auto lmul = lhs.multiplier();
    auto lpow = lhs.power();
    auto rmul = rhs.multiplier();
    auto rpow = rhs.power();

    return (lpow <= rpow) ? (lmul < Dyadic::shift(rmul, rpow - lpow))
                          : Dyadic::shift(lmul, lpow - rpow) < rmul;
}
bool rpy::intervals::operator<=(const Dyadic& lhs, const Dyadic& rhs)
{
    auto lmul = lhs.multiplier();
    auto lpow = lhs.power();
    auto rmul = rhs.multiplier();
    auto rpow = rhs.power();

    return (lpow <= rpow) ? (lmul <= Dyadic::shift(rmul, rpow - lpow))
                          : Dyadic::shift(lmul, lpow - rpow) <= rmul;
}
bool rpy::intervals::operator>(const Dyadic& lhs, const Dyadic& rhs)
{
    auto lmul = lhs.multiplier();
    auto lpow = lhs.power();
    auto rmul = rhs.multiplier();
    auto rpow = rhs.power();

    return (lpow <= rpow) ? (lmul > Dyadic::shift(rmul, rpow - lpow))
                          : Dyadic::shift(lmul, lpow - rpow) > rmul;
}
bool rpy::intervals::operator>=(const Dyadic& lhs, const Dyadic& rhs)
{
    auto lmul = lhs.multiplier();
    auto lpow = lhs.power();
    auto rmul = rhs.multiplier();
    auto rpow = rhs.power();

    return (lpow <= rpow) ? (lmul >= Dyadic::shift(rmul, rpow - lpow))
                          : Dyadic::shift(lmul, lpow - rpow) >= rmul;
}
std::ostream& rpy::intervals::operator<<(std::ostream& os, const Dyadic& arg)
{
    return os << '(' << arg.multiplier() << ", " << arg.power() << ')';
}
bool rpy::intervals::dyadic_equals(const Dyadic& lhs, const Dyadic& rhs)
{
    return lhs.power() == rhs.power() && lhs.multiplier() == rhs.multiplier();
}
bool rpy::intervals::rational_equals(const Dyadic& lhs, const Dyadic& rhs)
{
    Dyadic::multiplier_t ratio;
    if (lhs.multiplier() % rhs.multiplier() == 0
        && (ratio = lhs.multiplier() / rhs.multiplier()) >= 1) {
        Dyadic::power_t rel_tolerance = lhs.power() - rhs.power();
        if (rel_tolerance < 0) { return false; }
        return ratio == Dyadic::int_two_to_int_power(rel_tolerance);
    } else if (rhs.multiplier() % lhs.multiplier() == 0
               && (ratio = rhs.multiplier() / lhs.multiplier()) >= 1) {
        Dyadic::power_t rel_tolerance = rhs.power() - lhs.power();
        if (rel_tolerance < 0) { return false; }
        return ratio == Dyadic::int_two_to_int_power(rel_tolerance);
    }
    return false;
}

#define RPY_SERIAL_IMPL_CLASSNAME rpy::intervals::Dyadic
#include <roughpy/platform/serialization_instantiations.inl>
