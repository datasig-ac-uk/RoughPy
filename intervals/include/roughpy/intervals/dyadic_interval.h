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

#ifndef ROUGHPY_INTERVALS_DYADIC_INTERVAL_H_
#define ROUGHPY_INTERVALS_DYADIC_INTERVAL_H_

#include "dyadic.h"
#include "interval.h"
#include <roughpy/core/types.h>
#include <roughpy/platform/serialization.h>

#include <iosfwd>
#include <vector>

namespace rpy {
namespace intervals {

class RPY_EXPORT DyadicInterval : public Dyadic, public Interval
{
    using Dyadic::m_multiplier;
    using Dyadic::m_power;

protected:
    using Interval::m_interval_type;

public:
    using typename Dyadic::multiplier_t;
    using typename Dyadic::power_t;

    using Dyadic::Dyadic;

    DyadicInterval() = default;

    explicit DyadicInterval(IntervalType itype) : Dyadic(), Interval(itype)
    {
        RPY_DBG_ASSERT(itype == IntervalType::Clopen
                       || itype == IntervalType::Opencl);
    }
    explicit DyadicInterval(Dyadic dyadic) : Dyadic(dyadic), Interval() {}
    DyadicInterval(Dyadic dyadic, IntervalType itype)
        : Dyadic(dyadic), Interval(itype)
    {}
    DyadicInterval(multiplier_t k, power_t n, IntervalType itype)
        : Dyadic(k, n), Interval(itype)
    {
        RPY_DBG_ASSERT(itype == IntervalType::Clopen
                       || itype == IntervalType::Opencl);
    }
    DyadicInterval(Dyadic dyadic, power_t resolution,
                   IntervalType itype = IntervalType::Clopen);
    DyadicInterval(param_t val, power_t resolution,
                   IntervalType itype = IntervalType::Clopen);

    RPY_NO_DISCARD
    multiplier_t unit() const noexcept
    {
        return (m_interval_type == IntervalType::Clopen) ? 1 : -1;
    }

    using Dyadic::operator++;
    using Dyadic::operator--;
    using Dyadic::multiplier;
    using Dyadic::power;

    RPY_NO_DISCARD Dyadic dincluded_end() const;
    RPY_NO_DISCARD Dyadic dexcluded_end() const;
    RPY_NO_DISCARD Dyadic dinf() const;
    RPY_NO_DISCARD Dyadic dsup() const;

    RPY_NO_DISCARD param_t inf() const override;
    RPY_NO_DISCARD param_t sup() const override;
    RPY_NO_DISCARD param_t included_end() const override;
    RPY_NO_DISCARD param_t excluded_end() const override;

    RPY_NO_DISCARD
    DyadicInterval shrink_to_contained_end(power_t arg = 1) const;
    RPY_NO_DISCARD
    DyadicInterval shrink_to_omitted_end() const;
    DyadicInterval& shrink_interval_right();
    DyadicInterval& shrink_interval_left(power_t arg = 1);
    DyadicInterval& expand_interval(power_t arg = 1);

    RPY_NO_DISCARD
    bool contains_dyadic(const DyadicInterval& other) const;
    RPY_NO_DISCARD
    bool aligned() const;

    DyadicInterval& flip_interval();

    RPY_NO_DISCARD
    DyadicInterval shift_forward(multiplier_t arg = 1) const;
    RPY_NO_DISCARD
    DyadicInterval shift_back(multiplier_t arg = 1) const;

    DyadicInterval& advance() noexcept;

    RPY_SERIAL_SERIALIZE_FN();
};

RPY_EXPORT
std::ostream& operator<<(std::ostream& os, const DyadicInterval& di);

RPY_EXPORT
std::vector<DyadicInterval>
to_dyadic_intervals(const Interval& interval, typename Dyadic::power_t tol,
                    IntervalType itype = IntervalType::Clopen);

RPY_EXPORT
bool operator<(const DyadicInterval& lhs, const DyadicInterval& rhs) noexcept;

RPY_SERIAL_SERIALIZE_FN_IMPL(DyadicInterval)
{
    RPY_SERIAL_SERIALIZE_BASE(Dyadic);
    RPY_SERIAL_SERIALIZE_NVP("type", m_interval_type);
}

}// namespace intervals
}// namespace rpy

#endif// ROUGHPY_INTERVALS_DYADIC_INTERVAL_H_
