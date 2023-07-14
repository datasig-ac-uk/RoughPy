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

#ifndef ROUGHPY_INTERVALS_REAL_INTERVAL_H_
#define ROUGHPY_INTERVALS_REAL_INTERVAL_H_

#include "interval.h"

#include <roughpy/platform/serialization.h>

#include <utility>

namespace rpy {
namespace intervals {

class RPY_EXPORT RealInterval : public Interval
{
    param_t m_inf = 0.0;
    param_t m_sup = 1.0;

public:
    RealInterval() = default;
    RealInterval(const RealInterval&) = default;
    RealInterval(RealInterval&&) noexcept = default;

    RealInterval& operator=(const RealInterval&) = default;
    RealInterval& operator=(RealInterval&&) noexcept = default;

    RealInterval(
            param_t inf, param_t sup, IntervalType itype = IntervalType::Clopen
    )
        : Interval(itype), m_inf(inf), m_sup(sup)
    {
        if (m_inf > m_sup) { std::swap(m_inf, m_sup); }
    }

    explicit RealInterval(const Interval& interval)
        : Interval(interval.type()), m_inf(interval.inf()),
          m_sup(interval.sup())
    {}

    explicit RealInterval(const Interval& interval, IntervalType itype)
        : Interval(itype), m_inf(interval.inf()), m_sup(interval.sup())
    {}

    static RealInterval unbounded() noexcept
    {
        return {-std::numeric_limits<param_t>::infinity(),
                std::numeric_limits<param_t>::infinity()};
    }

    static RealInterval left_unbounded(
            param_t sup = 0.0, IntervalType itype = IntervalType::Clopen
    ) noexcept
    {
        return {-std::numeric_limits<param_t>::infinity(), sup, itype};
    }

    static RealInterval right_unbounded(
            param_t inf = 0.0, IntervalType itype = IntervalType::Clopen
    ) noexcept
    {
        return {inf, std::numeric_limits<param_t>::infinity(), itype};
    }

    RPY_NO_DISCARD param_t inf() const override { return m_inf; }
    RPY_NO_DISCARD param_t sup() const override { return m_sup; }

    RPY_NO_DISCARD bool contains(const Interval& arg) const noexcept override;

    RPY_SERIAL_SERIALIZE_FN();
};

RPY_SERIAL_SERIALIZE_FN_IMPL(RealInterval)
{
    RPY_SERIAL_SERIALIZE_NVP("type", m_interval_type);
    RPY_SERIAL_SERIALIZE_NVP("inf", m_inf);
    RPY_SERIAL_SERIALIZE_NVP("sup", m_sup);
}

}// namespace intervals
}// namespace rpy

#endif// ROUGHPY_INTERVALS_REAL_INTERVAL_H_
