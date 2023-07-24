// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
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
// Created by user on 05/07/23.
//
#include "partition.h"

#include <algorithm>
#include <cmath>
#include <limits>

using namespace rpy;
using namespace rpy::intervals;

Partition::Partition(RealInterval base) : RealInterval(std::move(base)) {}

Partition::Partition(RealInterval base, Slice<param_t> intermediate_points)
    : RealInterval(std::move(base))
{
    m_intermediate_points.reserve(intermediate_points.size());
    for (auto&& mid : intermediate_points) {
        if (contains_point(mid) && mid != inf() && mid != sup()) {
            m_intermediate_points.push_back(mid);
        }
    }
    std::sort(m_intermediate_points.begin(), m_intermediate_points.end());
}
Partition Partition::refine_midpoints() const
{
    // Only copy the base interval, we'll fill in the points
    Partition result(static_cast<const RealInterval&>(*this));
    result.m_intermediate_points.reserve(2 * size());

    param_t prev = inf();
    for (const auto& old_intermediate : m_intermediate_points) {
        auto mid = (old_intermediate + prev) / 2.0;
        result.m_intermediate_points.push_back(mid);
        result.m_intermediate_points.push_back(old_intermediate);
        prev = old_intermediate;
    }
    result.m_intermediate_points.push_back((sup() + prev) / 2.0);

    return result;
}

param_t Partition::mesh() const noexcept
{
    auto prev = inf();
    auto diff = std::numeric_limits<param_t>::infinity();

    for (const auto& intermediate : m_intermediate_points) {
        auto new_diff = std::abs(intermediate - prev);
        if (new_diff < diff) { diff = new_diff; }
        prev = intermediate;
    }
    {
        auto new_diff = std::abs(sup() - prev);
        if (new_diff < diff) { diff = new_diff; }
    }

    return diff;
}

RealInterval Partition::operator[](dimn_t i) const
{
    const auto num_midpoints = m_intermediate_points.size();
    RPY_CHECK(i <= num_midpoints);

    if (num_midpoints == 0) {
        // Special case, there are no intermediate points, so the 0th interval
        // is the whole interval
        return RealInterval(*this);
    }

    if (i == 0) {
        // Left-most interval is [inf(), first_midpoint)
        return RealInterval(inf(), m_intermediate_points.front(), type());
    }
    if (i == num_midpoints) {
        // Right-most interval is [last_midpoint, sup())
        return RealInterval(m_intermediate_points.back(), sup(), type());
    }

    // Otherwise the end-points are both intermediate points
    // Since the "0th" index is the left-most, the 1st index is between the
    // 0th intermediate point and the 1st intermediate point
    return RealInterval(m_intermediate_points[i - 1], m_intermediate_points[i],
                        type());
}

void Partition::insert_intermediate(param_t new_intermediate)
{
    if (!contains_point(new_intermediate)) {
        RPY_THROW(std::invalid_argument,"provided intermediate does not lie "
                                    "within the interval");
    }

    // Use find-insert rather than push_back-sort since this should
    // require fewer comparisons
    auto insert_point
            = std::lower_bound(m_intermediate_points.begin(),
                               m_intermediate_points.end(), new_intermediate);
    m_intermediate_points.insert(insert_point, new_intermediate);
}

Partition Partition::merge(const Partition& other) const
{
    RPY_CHECK(type() == other.type());

    const auto linf = inf();
    const auto lsup = sup();
    const auto rinf = other.inf();
    const auto rsup = other.sup();

    Partition result(
            RealInterval(std::min(linf, rinf), std::max(lsup, rsup), type()));

    if (lsup < rinf) {
        result.m_intermediate_points.reserve(2);
        if RPY_LIKELY (lsup != linf) {
            result.m_intermediate_points.push_back(lsup);
        }
        if RPY_LIKELY (rinf != rsup) {
            result.m_intermediate_points.push_back(rinf);
        }
    } else if (rsup < linf) {
        result.m_intermediate_points.reserve(2);
        if RPY_LIKELY (rinf != rsup) {
            result.m_intermediate_points.push_back(rsup);
        }
        if RPY_LIKELY (lsup != linf) {
            result.m_intermediate_points.push_back(linf);
        }
    } else {

        // At worst, the result has size size() + other.size();
        result.m_intermediate_points.reserve(size() + other.size());

        auto lit = m_intermediate_points.cbegin();
        const auto lend = m_intermediate_points.cend();
        auto rit = other.m_intermediate_points.cbegin();
        const auto rend = other.m_intermediate_points.cend();

        auto max_inf = std::max(linf, rinf);
        auto min_sup = std::min(lsup, rsup);
        // First handle the cases where *lit < rinf and *rit < linf
        // We know that all *lit > linf and *rit > rinf, so the insertion
        // is linear between inf and max(linf, rinf)
        // Only one of these loops will execute
        for (; lit != lend && *lit < rinf; ++lit) {
            result.m_intermediate_points.push_back(*lit);
        }
        for (; rit != rend && *rit < linf; ++rit) {
            result.m_intermediate_points.push_back(*rit);
        }

        /*
         * Here we need to be careful to make sure we don't insert max(linf,
         * rinf) twice, in the case where linf == *rit or rinf == *lit. Also,
         * only do this if linf != rinf
         */
        if (linf != rinf) {
            result.m_intermediate_points.push_back(max_inf);
            if (rit != rend && *rit == max_inf) { ++rit; }
            if (lit != lend && *lit == max_inf) { ++lit; }
        }

        /*
         * Now comes the interesting bit. We need to interleave between
         * max(linf, rinf) and min(lsup, rsup). If lsup < rsup then it is never
         * true that *lit >= rsup. Similarly, if rsup < lsup then it is never
         * true that *rit >= lsup. Consequently, we only need to check that
         * lit and rit don't hit lend and rend (respectively) in order to know
         * that we've got to min(lsup, rsup).
         */
        while (lit != lend && rit != rend) {
            if (*lit == *rit) {
                result.m_intermediate_points.push_back(*lit);
                ++lit;
                ++rit;
            } else if (*lit < *rit) {
                result.m_intermediate_points.push_back(*lit);
                ++lit;
            } else {
                result.m_intermediate_points.push_back(*rit);
                ++rit;
            }
        }

        /*
         * Insert any *lit or *rit that lie between lsup and rsup or rsup and
         * lsup respectively.
         */
        for (; lit != lend && *lit < rsup; ++lit) {
            result.m_intermediate_points.push_back(*lit);
        }
        for (; rit != rend && *rit < lsup; ++rit) {
            result.m_intermediate_points.push_back(*rit);
        }

        /*
         * Now insert min(lsup, rsup). Again we need to be careful that we don't
         * accidentally insert the same value twice if it appears as both an
         * end point and as an intermediate point.
         * Only do this step if lsup != rsup and if min(lsup, rsup) != max(linf,
         * rinf).
         */
        if (lsup != rsup && min_sup != max_inf) {
            result.m_intermediate_points.push_back(min_sup);
            if (lit != lend && *lit == min_sup) { ++lit; }
            if (rit != rend && *rit == min_sup) { ++rit; }
        }

        // Now finish off whatever intermediate points remain.
        // Only one of these loops will execute
        for (; lit != lend; ++lit) {
            result.m_intermediate_points.push_back(*lit);
        }
        for (; rit != rend; ++rit) {
            result.m_intermediate_points.push_back(*rit);
        }
    }

    return result;
}
