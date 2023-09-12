// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
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
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//
// Created by user on 11/09/23.
//

#include "dyadic_merging_helper.h"

using namespace rpy;
using namespace intervals;

streams::IntegerRange
streams::DyadicMergingHelper::insert(intervals::RealInterval interval)
{
    const auto range = interval_to_range(interval);

    auto it = m_ranges.begin();
    auto end = m_ranges.end();

    // First get to the last interval whose lower bound is smaller than range
    for (; it != end && it->lower < range.lower; ++it) {}

    /*
     * At this point, either lit == end OR lit->lower >= range.lower.
     * If lit = end, then insert a new entry and return
     */
    if (it == end) {
        m_ranges.insert(it, range);
        return range;
    }

    /*
     * If lit->lower > range.upper then there is no overlap insert new entry
     * and return
     */
    if (it->lower > range.upper) {
        m_ranges.insert(it, range);
        return range;
    }

    /*
     * Finally, we're in the case where lit->lower <= range.upper so there is an
     * overlap, so we need to absorb lit and any following ranges for which
     * there is an overlap.
     */
    auto entry = it++;
    entry->lower = range.lower;

    while (it != end && it->lower <= range.upper) {
        entry->upper = it->upper;
        it = m_ranges.erase(it);
    }

    return range;
}
std::vector<intervals::DyadicInterval>
streams::DyadicMergingHelper::to_dyadics()
{
    std::vector<DyadicInterval> result;

    dimn_t num_dyadics = 0;
    for (const auto& range : m_ranges) {
        num_dyadics += range.upper - range.lower;
    }

    result.reserve(num_dyadics);
    for (const auto& range : m_ranges) {
        for (auto i=range.lower; i<range.upper; ++i) {
            result.emplace_back(i, m_resolution);
        }
    }

    return result;
}

dyadic_multiplier_t
streams::DyadicMergingHelper::offset_to(dyadic_multiplier_t val)
{
    RPY_CHECK(!m_ranges.empty());

    dyadic_multiplier_t result = 0;

    auto it = m_ranges.begin();
    auto itm1 = it++;
    const auto end = m_ranges.end();

    for (; it != end && val < it->lower; ++it, ++itm1) {
        result += it->lower - itm1->upper;
    }

    return result;
}
