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

#include <roughpy/streams/compute_tree.h>

#include "dyadic_merging_helper.h"

using namespace rpy;
using namespace rpy::streams;
using namespace rpy::intervals;


DyadicComputationTree::DyadicComputationTree(
        const std::vector<intervals::RealInterval>& queries,
        resolution_t dyadic_resolution
)
    : m_queries(queries), m_resolution(dyadic_resolution)
{
    std::vector<IntegerRange> ranges;
    ranges.reserve(m_queries.size());

    DyadicMergingHelper helper(m_resolution);

    for (const auto& query : queries) {
        ranges.push_back(helper.insert(query));
    }

    m_dyadics = helper.to_dyadics();

    auto qit = queries.begin();
    auto rit = ranges.begin();
    const auto rend = ranges.end();

    const auto begin = m_dyadics.cbegin();
    for (; rit != rend; ++rit, ++qit) {
        Entry& e = m_mapping[*qit];
        e.begin = begin + helper.offset_to(rit->lower);
        e.end = e.begin + rit->upper - rit->lower;
    }

}
