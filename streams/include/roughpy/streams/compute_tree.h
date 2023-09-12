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

#ifndef ROUGHPY_STREAMS_COMPUTE_TREE_H_
#define ROUGHPY_STREAMS_COMPUTE_TREE_H_

#include <roughpy/core/flat_tree.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>

#include <roughpy/platform/serialization.h>

#include <roughpy/algebra/lie.h>

#include <roughpy/intervals/dyadic_interval.h>
#include <roughpy/intervals/interval.h>
#include <roughpy/intervals/real_interval.h>

#include <boost/container_hash/hash.hpp>

#include <vector>

namespace rpy {
namespace streams {

class DyadicComputationTree
{
    using dyadic_list_t = std::vector<intervals::DyadicInterval>;
    using dyadic_iterator = typename dyadic_list_t::const_iterator;

    std::vector<intervals::DyadicInterval> m_dyadics;

    struct Entry {
        dyadic_iterator begin;
        dyadic_iterator end;
    };

    using mapping_t = std::unordered_map<
            intervals::RealInterval,
            Entry,
            boost::hash<intervals::RealInterval>>;

    const std::vector<intervals::RealInterval>& m_queries;
    mapping_t m_mapping;
    resolution_t m_resolution;


    dyadic_multiplier_t granular_distance(param_t begin, param_t end) const
            noexcept {
        return (intervals::DyadicInterval(end, m_resolution).multiplier()
                - intervals::DyadicInterval(begin, m_resolution).multiplier());
    }

public:
    explicit DyadicComputationTree(
            const std::vector<intervals::RealInterval>& queries,
            resolution_t dyadic_resolution
    );



};

}// namespace streams
}// namespace rpy

#endif// ROUGHPY_STREAMS_COMPUTE_TREE_H_
