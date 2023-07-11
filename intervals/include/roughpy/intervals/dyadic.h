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

#ifndef ROUGHPY_INTERVALS_DYADIC_H_
#define ROUGHPY_INTERVALS_DYADIC_H_

#include <roughpy/core/types.h>
#include <roughpy/platform/serialization.h>

#include <cassert>
#include <cmath>
#include <iosfwd>
#include <limits>

namespace rpy {
namespace intervals {

class RPY_EXPORT Dyadic
{
public:
    using multiplier_t = dyadic_multiplier_t;
    using power_t = dyadic_depth_t;

protected:
    multiplier_t m_multiplier = 0;
    power_t m_power = 0;

public:
    RPY_NO_DISCARD
    static constexpr multiplier_t mod(multiplier_t a, multiplier_t b)
    {
        multiplier_t r = a % b;
        return (r < 0) ? (r + abs(b)) : r;
    }
    RPY_NO_DISCARD
    static constexpr multiplier_t int_two_to_int_power(power_t exponent)
    {
        RPY_DBG_ASSERT(exponent >= 0);
        return multiplier_t(1) << exponent;
    }
    RPY_NO_DISCARD
    static constexpr multiplier_t shift(multiplier_t k, power_t n)
    {
        return k * int_two_to_int_power(n);
    }

    Dyadic() = default;

    explicit Dyadic(multiplier_t k, power_t n = 0) : m_multiplier(k), m_power(n)
    {}

    RPY_NO_DISCARD
    multiplier_t multiplier() const noexcept { return m_multiplier; }
    RPY_NO_DISCARD
    power_t power() const noexcept { return m_power; }

    RPY_NO_DISCARD
    explicit operator param_t() const noexcept;

    Dyadic& move_forward(multiplier_t arg);
    Dyadic& operator++();
    RPY_NO_DISCARD
    const Dyadic operator++(int);
    Dyadic& operator--();
    RPY_NO_DISCARD
    const Dyadic operator--(int);

    bool rebase(power_t resolution = std::numeric_limits<power_t>::lowest());

private:
    RPY_SERIAL_ACCESS();
    RPY_SERIAL_SERIALIZE_FN();
};

RPY_EXPORT
bool operator<(const Dyadic& lhs, const Dyadic& rhs);

RPY_EXPORT
bool operator<=(const Dyadic& lhs, const Dyadic& rhs);

RPY_EXPORT
bool operator>(const Dyadic& lhs, const Dyadic& rhs);

RPY_EXPORT
bool operator>=(const Dyadic& lhs, const Dyadic& rhs);

RPY_EXPORT
std::ostream& operator<<(std::ostream& os, const Dyadic& arg);

RPY_EXPORT
bool dyadic_equals(const Dyadic& lhs, const Dyadic& rhs);

RPY_EXPORT
bool rational_equals(const Dyadic& lhs, const Dyadic& rhs);

RPY_SERIAL_SERIALIZE_FN_IMPL(Dyadic)
{
    RPY_SERIAL_SERIALIZE_NVP("multiplier", m_multiplier);
    RPY_SERIAL_SERIALIZE_NVP("power", m_power);
}

}// namespace intervals
}// namespace rpy

#endif// ROUGHPY_INTERVALS_DYADIC_H_
