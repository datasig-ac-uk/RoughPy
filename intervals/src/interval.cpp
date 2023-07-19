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

#include <ostream>
#include <roughpy/intervals/interval.h>

using namespace rpy;
using namespace rpy::intervals;

param_t Interval::included_end() const
{
    if (m_interval_type == IntervalType::Clopen) { return inf(); }
    if (m_interval_type == IntervalType::Opencl) { return sup(); }
    RPY_THROW(std::runtime_error,
            "included_end is not valid for intervals that are not half open");
}
param_t Interval::excluded_end() const
{
    if (m_interval_type == IntervalType::Clopen) { return sup(); }
    if (m_interval_type == IntervalType::Opencl) { return inf(); }
    RPY_THROW(std::runtime_error,
            "excluded_end is not valid for intervals that are not half open");
}
bool Interval::contains_point(param_t arg) const noexcept
{
    if (m_interval_type == IntervalType::Clopen) {
        return inf() <= arg && arg < sup();
    }
    if (m_interval_type == IntervalType::Opencl) {
        return inf() < arg && arg <= sup();
    }

    return false;
}
bool Interval::is_associated(const Interval& arg) const noexcept
{
    return contains_point(arg.included_end());
}
bool Interval::contains(const Interval& arg) const noexcept
{
    return contains_point(arg.inf()) && contains_point(arg.sup());
}
bool Interval::intersects_with(const Interval& arg) const noexcept
{
    auto lhs_inf = inf();
    auto lhs_sup = sup();
    auto rhs_inf = arg.inf();
    auto rhs_sup = arg.sup();

    if ((lhs_inf <= rhs_inf && lhs_sup > rhs_inf)
        || (rhs_inf <= lhs_inf && rhs_sup > lhs_inf)) {
        // [l--[r---l)--r) || [r--[l--r)--l)
        return true;
    }
    if (rhs_inf == lhs_sup) {
        // (l--l][r--r)
        return m_interval_type == IntervalType::Opencl
                && arg.m_interval_type == IntervalType::Clopen;
    }
    if (lhs_inf == rhs_sup) {
        // (r--r][l---l)
        return m_interval_type == IntervalType::Clopen
                && arg.m_interval_type == IntervalType::Opencl;
    }
    return false;
}
bool Interval::operator==(const Interval& other) const
{
    return m_interval_type == other.m_interval_type && inf() == other.inf()
            && sup() == other.sup();
}
bool Interval::operator!=(const Interval& other) const
{
    return !operator==(other);
}

std::ostream& rpy::intervals::operator<<(std::ostream& os,
                                         const Interval& interval)
{
    if (interval.type() == IntervalType::Clopen) {
        os << '[';
    } else {
        os << '(';
    }

    os << interval.inf() << ", " << interval.sup();

    if (interval.type() == IntervalType::Clopen) {
        os << ')';
    } else {
        os << ']';
    }

    return os;
}
