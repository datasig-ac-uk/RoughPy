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

/**
 * @brief The IntervalType enum class represents the different types of
 * intervals.
 */
enum class IntervalType : uint32_t
{
    Clopen,
    Opencl
};

/**
 * @class Interval
 * @brief The Interval class represents an abstract interval.
 *
 * This class provides a base implementation for interval operations and
 * defines common interface for concrete interval subclasses to implement.
 */
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

    /**
     * @brief Returns the lower bound of the interval.
     *
     * @details This method returns the lower bound of the interval.
     *
     * @return The lower bound of the interval.
     */
    RPY_NO_DISCARD virtual param_t inf() const = 0;
    /**
     * @brief Returns the upper bound of the interval.
     *
     * This method returns the upper bound of the interval.
     *
     * @return The upper bound of the interval.
     */
    RPY_NO_DISCARD virtual param_t sup() const = 0;

    /**
     * @brief Returns the included end of the interval.
     *
     * This method returns the included end of the interval. If the interval
     * type is Clopen, it returns the upper bound of the interval. If the
     * interval type is Opencl, it returns the lower bound of the interval.
     *
     * @return The included end of the interval.
     */
    RPY_NO_DISCARD virtual param_t included_end() const;
    /**
     * @brief Returns the excluded end of the interval.
     *
     * This method returns the excluded end of the interval. If the interval
     * type is Clopen, it returns the upper bound of the interval. If the
     * interval type is Opencl, it returns the lower bound of the interval.
     *
     * @return The excluded end of the interval.
     */
    RPY_NO_DISCARD virtual param_t excluded_end() const;

    /**
     * @brief Checks if a given point is contained within the interval.
     *
     * @param arg The point to be checked.
     * @return Returns true if the point is contained within the interval, false
     * otherwise.
     * @note This method considers the boundaries of the interval depending on
     * its type.
     */
    RPY_NO_DISCARD virtual bool contains_point(param_t arg) const noexcept;
    /**
     * @brief Checks whether the given interval is associated with the current
     * interval.
     *
     * This method determines whether the given interval is associated with the
     * current interval by checking if the included end point of the given
     * interval is contained in the current interval.
     *
     * @param arg The interval to check for association.
     * @return True if the given interval is associated with the current
     * interval, false otherwise.
     *
     * @note This method does not modify the current interval.
     * @note This method assumes that the given interval is not null.
     * @note This method is noexcept, meaning it does not throw any exceptions.
     */
    RPY_NO_DISCARD virtual bool is_associated(const Interval& arg) const noexcept;
    /**
     * @brief Check if the given interval is contained within this interval.
     *
     * This method checks if the given interval @p arg is completely contained
     * within this interval. It returns true if both the lower and upper bound
     * of @p arg are within the range of this interval, otherwise it returns
     * false.
     *
     * @param arg The interval to check for containment.
     * @return True if the given interval is contained within this interval,
     *         false otherwise.
     */
    RPY_NO_DISCARD virtual bool contains(const Interval& arg) const noexcept;
    /**
     * @brief Checks whether the given interval intersects with the current
     * interval.
     *
     * This method determines whether the given interval intersects with the
     * current interval by checking if there is any overlap between the two
     * intervals.
     *
     * @param arg The interval to check for intersection.
     * @return True if the given interval intersects with the current interval,
     * false otherwise.
     * @note This method does not modify the current interval.
     * @note This method is noexcept, meaning it does not throw any exceptions.
     */
    RPY_NO_DISCARD virtual bool intersects_with(const Interval& arg) const noexcept;

    RPY_NO_DISCARD
    virtual bool operator==(const Interval& other) const;
    RPY_NO_DISCARD
    virtual bool operator!=(const Interval& other) const;

    RPY_SERIAL_SERIALIZE_FN();

};

ROUGHPY_INTERVALS_EXPORT
std::ostream& operator<<(std::ostream& os, const Interval& interval);


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
