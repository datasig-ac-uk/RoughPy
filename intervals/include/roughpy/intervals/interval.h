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

#ifndef ROUGHPY_INTERVALS_INTERVAL_H_
#define ROUGHPY_INTERVALS_INTERVAL_H_

#include <roughpy/core/types.h>
#include <roughpy/platform/serialization.h>

#include <ostream>

#include <roughpy/platform/serialization.h>

#include "roughpy_intervals_export.h"

namespace rpy {
namespace intervals {


class RealInterval;

enum class IntervalType : uint32_t
{
    Clopen,
    Opencl
};

class ROUGHPY_INTERVALS_EXPORT Interval
{
protected:
    IntervalType m_interval_type = IntervalType::Clopen;

public:
    Interval() = default;

    explicit Interval(IntervalType itype) : m_interval_type(itype) {}

    virtual ~Interval() = default;

    RPY_NO_DISCARD
    inline IntervalType type() const noexcept { return m_interval_type; }

    RPY_NO_DISCARD
    virtual param_t inf() const = 0;
    RPY_NO_DISCARD
    virtual param_t sup() const = 0;

    RPY_NO_DISCARD
    virtual param_t included_end() const;
    RPY_NO_DISCARD
    virtual param_t excluded_end() const;

    RPY_NO_DISCARD
    virtual bool contains_point(param_t arg) const noexcept;
    RPY_NO_DISCARD
    virtual bool is_associated(const Interval& arg) const noexcept;
    RPY_NO_DISCARD
    virtual bool contains(const Interval& arg) const noexcept;
    RPY_NO_DISCARD
    virtual bool intersects_with(const Interval& arg) const noexcept;

    RPY_NO_DISCARD
    virtual bool operator==(const Interval& other) const;
    RPY_NO_DISCARD
    virtual bool operator!=(const Interval& other) const;

    RPY_SERIAL_SERIALIZE_FN();

};

ROUGHPY_INTERVALS_EXPORT
std::ostream& operator<<(std::ostream& os, const Interval& interval);


/**
 * @brief Computes the intersection of two intervals.
 *
 * @param lhs The first interval to be used in the intersection operation.
 * @param rhs The second interval to be used in the intersection operation.
 *
 * @return A new interval describing the intersection of lhs and rhs.
 *         The result is degenerate if lhs and rhs do not intersect.
 */
RPY_NO_DISCARD ROUGHPY_INTERVALS_EXPORT
RealInterval intersection(const Interval& lhs, const Interval& rhs) noexcept;

/**
 * @brief Computes the interval union of two intervals.
 *
 * The interval union is defined as the smallest interval that contains both
 * arguments.
 *
 * @param lhs The first interval to be used in the union operation.
 * @param rhs The second interval to be used in the union operation.
 *
 * @return A new interval representing the union of lhs and rhs.
 *         The result encompasses all points contained in either lhs or rhs.
 */
RPY_NO_DISCARD ROUGHPY_INTERVALS_EXPORT
RealInterval interval_union(const Interval& lhs, const Interval& rhs) noexcept;


RPY_SERIAL_LOAD_FN_EXT(IntervalType) {
    uint32_t tmp;
    RPY_SERIAL_SERIALIZE_BARE(tmp);
    value = static_cast<IntervalType>(tmp);
}
RPY_SERIAL_SAVE_FN_EXT(IntervalType) {
    RPY_SERIAL_SERIALIZE_BARE(static_cast<uint32_t>(value));
}

RPY_SERIAL_SERIALIZE_FN_IMPL(Interval) {
    RPY_SERIAL_SERIALIZE_NVP("type", m_interval_type);
}

}// namespace intervals
}// namespace rpy

RPY_SERIAL_SPECIALIZE_TYPES(rpy::intervals::IntervalType,
                            rpy::serial::specialization::non_member_load_save);

#endif// ROUGHPY_INTERVALS_INTERVAL_H_
