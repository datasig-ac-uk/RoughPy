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
// Created by user on 03/03/23.
//

#ifndef ROUGHPY_INTERVALS_SRC_DYADIC_SEARCHER_H
#define ROUGHPY_INTERVALS_SRC_DYADIC_SEARCHER_H

#include <roughpy/intervals/segmentation.h>

#include <cassert>
#include <deque>
#include <map>
#include <utility>

#include <roughpy/intervals/dyadic.h>

#include "scaled_predicate.h"

namespace rpy {
namespace intervals {

struct DyadicRealStrictLess {
    bool operator()(const Dyadic& lhs, const Dyadic& rhs) const noexcept
    {
        auto max = std::max(lhs.power(), rhs.power());
        return (lhs.multiplier() << (max - lhs.power()))
                < (rhs.multiplier() << (max - rhs.power()));
    }
};

struct DyadicRealStrictGreater {
    bool operator()(const Dyadic& lhs, const Dyadic& rhs) const noexcept
    {
        auto max = std::max(lhs.power(), rhs.power());
        return (lhs.multiplier() << (max - lhs.power()))
                > (rhs.multiplier() << (max - rhs.power()));
    }
};

class DyadicSearcher
{
    predicate_t m_predicate;
    std::map<Dyadic, Dyadic, DyadicRealStrictGreater> m_seen;
    dyadic_depth_t m_max_depth;

protected:
    void expand_left(ScaledPredicate& predicate,
                     std::deque<DyadicInterval>& current) const;
    void expand_right(ScaledPredicate& predicate,
                      std::deque<DyadicInterval>& current) const;
    void expand(ScaledPredicate& predicate, DyadicInterval found_interval);

public:
    DyadicSearcher(predicate_t&& predicate, dyadic_depth_t max_depth)
        : m_predicate(std::move(predicate)), m_max_depth(max_depth)
    {}

private:
    ScaledPredicate rescale_to_unit_interval(const Interval& original);
    void get_next_dyadic(DyadicInterval& current) const;
    std::vector<RealInterval> find_in_unit_interval(ScaledPredicate& predicate);

public:
    std::vector<RealInterval> operator()(const Interval& original);
};

}// namespace intervals
}// namespace rpy

#endif// ROUGHPY_INTERVALS_SRC_DYADIC_SEARCHER_H
