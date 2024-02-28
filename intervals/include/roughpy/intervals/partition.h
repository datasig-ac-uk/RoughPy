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

#ifndef ROUGHPY_INTERVALS_PARTITION_H_
#define ROUGHPY_INTERVALS_PARTITION_H_

#include "real_interval.h"

#include <roughpy/core/macros.h>
#include <roughpy/core/slice.h>
#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>
#include <roughpy/platform/serialization.h>

#include "roughpy_intervals_export.h"

namespace rpy {
namespace intervals {

class ROUGHPY_INTERVALS_EXPORT Partition : public RealInterval
{
public:
    using intermediates_t = std::vector<param_t>;

private:
    intermediates_t m_intermediate_points;

public:
    Partition() = default;

    explicit Partition(RealInterval base);
    explicit Partition(RealInterval base, Slice<param_t> intermediate_points);

    Partition(RealInterval base, std::vector<param_t>&& intermediates)
        : RealInterval(std::move(base)),
          m_intermediate_points(std::move(intermediates))
    {}

    RPY_NO_DISCARD
    Partition refine_midpoints() const;

    RPY_NO_DISCARD
    param_t mesh() const noexcept;

    RPY_NO_DISCARD
    dimn_t size() const noexcept { return 1 + m_intermediate_points.size(); }

    RPY_NO_DISCARD
    RealInterval operator[](dimn_t i) const;

    const intermediates_t& intermediates() const noexcept
    {
        return m_intermediate_points;
    }

    void insert_intermediate(param_t new_intermediate);

    RPY_NO_DISCARD
    Partition merge(const Partition& other) const;


    RPY_NO_DISCARD
    bool operator=(const Partition& other) const noexcept {
        if (static_cast<const RealInterval&>(*this) != other) {
            return false;
        }
        if (m_intermediate_points.size() != other.m_intermediate_points.size()) {
            return false;
        }

        auto lit = m_intermediate_points.begin();
        auto lend = m_intermediate_points.end();
        auto rit = other.m_intermediate_points.begin();

        for (; lit != lend; ++lit, ++rit) {
            if (*lit != *rit) { return false; }
        }
        return true;
    }

    RPY_SERIAL_SERIALIZE_FN();
};

#ifdef RPY_COMPILING_INTERVALS
RPY_SERIAL_EXTERN_SERIALIZE_CLS_BUILD(Partition)
#else
RPY_SERIAL_EXTERN_SERIALIZE_CLS_IMP(Partition)
#endif

RPY_SERIAL_SERIALIZE_FN_IMPL(Partition) {
    RPY_SERIAL_SERIALIZE_BASE(RealInterval);
    RPY_SERIAL_SERIALIZE_NVP("intermediate_points", m_intermediate_points);
}


}// namespace intervals
}// namespace rpy

RPY_SERIAL_SPECIALIZE_TYPES(::rpy::intervals::Partition,
                            ::rpy::serial::specialization::member_serialize);

#endif// ROUGHPY_INTERVALS_PARTITION_H_
