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

#include <roughpy/intervals/dyadic_interval.h>

#include <algorithm>
#include <list>

#include <roughpy/intervals/real_interval.h>

using namespace rpy;
using namespace rpy::intervals;

DyadicInterval::DyadicInterval(Dyadic dyadic, Dyadic::power_t resolution,
                               IntervalType itype)
    : Dyadic(dyadic), Interval(itype)
{
    if (!rebase(resolution)) {
        multiplier_t k1 = m_multiplier;
        const multiplier_t one = unit();
        multiplier_t pow = int_two_to_int_power(m_power - resolution);
        m_multiplier = one * (k1 * one - mod(k1 * one, pow));
        bool is_int = rebase(resolution);
        RPY_CHECK(is_int);
    }
}
DyadicInterval::DyadicInterval(param_t val, Dyadic::power_t resolution,
                               IntervalType itype)
    : Dyadic(0, 0), Interval(itype)
{
    auto rescaled = ldexp(val, resolution);
    if (m_interval_type == IntervalType::Opencl) {
        m_multiplier = ceil(rescaled);
    } else {
        m_multiplier = floor(rescaled);
    }
    m_power = resolution;
}
Dyadic DyadicInterval::dincluded_end() const
{
    return static_cast<const Dyadic&>(*this);
}
Dyadic DyadicInterval::dexcluded_end() const
{
    return Dyadic(m_multiplier + unit(), m_power);
}
Dyadic DyadicInterval::dinf() const
{
    return m_interval_type == IntervalType::Clopen ? dincluded_end()
                                                   : dexcluded_end();
}
Dyadic DyadicInterval::dsup() const
{
    return m_interval_type == IntervalType::Clopen ? dexcluded_end()
                                                   : dincluded_end();
}
param_t DyadicInterval::inf() const { return static_cast<param_t>(dinf()); }
param_t DyadicInterval::sup() const { return static_cast<param_t>(dsup()); }
param_t DyadicInterval::included_end() const
{
    return static_cast<param_t>(dincluded_end());
}
param_t DyadicInterval::excluded_end() const
{
    return static_cast<param_t>(dexcluded_end());
}

DyadicInterval
DyadicInterval::shrink_to_contained_end(Dyadic::power_t arg) const
{
    return {static_cast<const Dyadic&>(*this), arg + m_power, m_interval_type};
}
DyadicInterval DyadicInterval::shrink_to_omitted_end() const
{
    return shrink_to_contained_end().flip_interval();
}
DyadicInterval& DyadicInterval::shrink_interval_right()
{
    if (m_interval_type == IntervalType::Opencl) {
        *this = shrink_to_contained_end();
    } else {
        *this = shrink_to_omitted_end();
    }
    return *this;
}
DyadicInterval& DyadicInterval::shrink_interval_left(Dyadic::power_t arg)
{
    RPY_DBG_ASSERT(arg >= 0);
    for (; arg > 0; --arg) {
        if (m_interval_type == IntervalType::Clopen) {
            *this = shrink_to_contained_end();
        } else {
            *this = shrink_to_omitted_end();
        }
    }
    return *this;
}
DyadicInterval& DyadicInterval::expand_interval(Dyadic::power_t arg)
{
    *this = DyadicInterval{dincluded_end(), m_power - arg};
    return *this;
}
bool DyadicInterval::contains_dyadic(const DyadicInterval& other) const
{
    if (other.m_interval_type != m_interval_type) { return false; }
    if (other.m_power >= m_power) {
        multiplier_t one(unit());
        multiplier_t pow = int_two_to_int_power(other.m_power - m_power);
        multiplier_t shifted = shift(m_multiplier, (other.m_power - m_power));
        multiplier_t aligned = one
                * (other.m_multiplier * one
                   - mod(other.m_multiplier * one, pow));

        return shifted == aligned;
    }
    return false;
}
bool DyadicInterval::aligned() const
{
    DyadicInterval parent{static_cast<const Dyadic&>(*this), m_power - 1,
                          m_interval_type};
    return operator==(
            DyadicInterval{static_cast<const Dyadic&>(parent), m_power});
}
DyadicInterval& DyadicInterval::flip_interval()
{
    if ((m_multiplier % 2) == 0) {
        m_multiplier += unit();
    } else {
        m_multiplier -= unit();
    }
    return *this;
}
DyadicInterval DyadicInterval::shift_forward(Dyadic::multiplier_t arg) const
{
    DyadicInterval tmp(*this);
    tmp.m_multiplier -= unit() * arg;
    return tmp;
}
DyadicInterval DyadicInterval::shift_back(Dyadic::multiplier_t arg) const
{
    DyadicInterval tmp(*this);
    tmp.m_multiplier += unit() * arg;
    return tmp;
}
DyadicInterval& DyadicInterval::advance() noexcept
{
    m_multiplier += unit();
    return *this;
}

std::ostream& rpy::intervals::operator<<(std::ostream& os,
                                         const DyadicInterval& di)
{
    return os << static_cast<const Interval&>(di);
}

std::vector<DyadicInterval>
rpy::intervals::to_dyadic_intervals(const Interval& interval,
                                    Dyadic::power_t tol, IntervalType itype)
{
    using iterator = std::list<DyadicInterval>::iterator;
    std::list<DyadicInterval> intervals;

    auto store_move = [&](DyadicInterval& b) {
        intervals.push_back(b.shrink_to_omitted_end());
        b.advance();
    };

    auto store_ = [&](iterator& p, DyadicInterval& e) -> iterator {
        return intervals.insert(p, e.shrink_to_contained_end());
    };

    RealInterval real{interval, itype};

    DyadicInterval begin{real.included_end(), tol, itype};
    DyadicInterval end{real.excluded_end(), tol, itype};

    while (!begin.contains_dyadic(end)) {
        auto next{begin};
        next.expand_interval();
        if (!begin.aligned()) { store_move(next); }
        begin = std::move(next);
    }

    auto p = intervals.end();
    for (auto next{end}; begin.contains_dyadic(next.expand_interval());) {
        if (!end.aligned()) { p = store_(p, next); }
        end = next;
    }

    if (itype == IntervalType::Opencl) { intervals.reverse(); }

    return {intervals.begin(), intervals.end()};
}

bool rpy::intervals::operator<(const DyadicInterval& lhs,
                               const DyadicInterval& rhs) noexcept
{
    if (lhs.type() != rhs.type()) { return false; }

    auto lhs_k = lhs.multiplier();
    auto lhs_n = lhs.power();
    auto rhs_k = rhs.multiplier();
    auto rhs_n = rhs.power();

    auto unit = lhs.unit();

    if (lhs_n == rhs_n) { return unit * (rhs_k - lhs_k) > 0; }

    if (lhs_n > rhs_n) {
        rhs_k = Dyadic::shift(rhs_k, lhs_n - rhs_n);
        return unit * (rhs_k - lhs_k) > 0;
    }

    lhs_k = Dyadic::shift(lhs_k, rhs_n - lhs_n);
    return unit * (rhs_k - lhs_k) >= 0;
}

#define RPY_SERIAL_IMPL_CLASSNAME rpy::intervals::DyadicInterval
#include <roughpy/platform/serialization_instantiations.inl>
