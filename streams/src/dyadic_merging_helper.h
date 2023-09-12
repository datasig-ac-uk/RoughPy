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

#ifndef ROUGHPY_STREAMS_SRC_DYADIC_MERGING_HELPER_H_
#define ROUGHPY_STREAMS_SRC_DYADIC_MERGING_HELPER_H_

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>
#include <roughpy/intervals/dyadic_interval.h>
#include <roughpy/intervals/real_interval.h>

#include <list>

namespace rpy {
namespace streams {

struct IntegerRange {
    dyadic_multiplier_t lower;
    dyadic_multiplier_t upper;
};

class DyadicMergingHelper
{
    std::list<IntegerRange> m_ranges;
    resolution_t m_resolution;

    IntegerRange interval_to_range(const intervals::RealInterval& ivl
    ) const noexcept
    {
        using intervals::DyadicInterval;
        return {DyadicInterval(ivl.inf(), m_resolution).multiplier(),
                // One past the end
                DyadicInterval(ivl.sup(), m_resolution).multiplier()
        };
    }

public:
    explicit DyadicMergingHelper(resolution_t resolution)
        : m_resolution(resolution)
    {}

    RPY_NO_DISCARD IntegerRange insert(intervals::RealInterval interval);

    RPY_NO_DISCARD std::vector<intervals::DyadicInterval> to_dyadics();

    RPY_NO_DISCARD dyadic_multiplier_t offset_to(dyadic_multiplier_t val);
};

}// namespace streams
}// namespace rpy

#endif// ROUGHPY_STREAMS_SRC_DYADIC_MERGING_HELPER_H_
