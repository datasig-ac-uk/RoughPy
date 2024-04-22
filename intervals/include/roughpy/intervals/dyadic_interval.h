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

#ifndef ROUGHPY_INTERVALS_DYADIC_INTERVAL_H_
#define ROUGHPY_INTERVALS_DYADIC_INTERVAL_H_

#include "dyadic.h"
#include "interval.h"
#include <roughpy/core/types.h>
#include <roughpy/core/container/vector.h>
#include <roughpy/platform/serialization.h>

#include <ostream>
#include <vector>

#include "roughpy_intervals_export.h"

namespace rpy {
namespace intervals {

/**
 * @class DyadicInterval
 * @brief Class representing a dyadic interval.
 *
 * This class is derived from both the Dyadic class and the Interval class,
 * and provides functionality for manipulating dyadic intervals.
 *
 * @see Dyadic
 * @see Interval
 */
class ROUGHPY_INTERVALS_EXPORT DyadicInterval : public Dyadic, public Interval
{
    using Dyadic::m_multiplier;
    using Dyadic::m_power;

protected:
    using Interval::m_interval_type;

public:
    using typename Dyadic::multiplier_t;
    using typename Dyadic::power_t;

    using Dyadic::Dyadic;

    DyadicInterval() = default;

    explicit DyadicInterval(IntervalType itype) : Dyadic(), Interval(itype)
    {
        RPY_DBG_ASSERT(
                itype == IntervalType::Clopen || itype == IntervalType::Opencl
        );
    }
    explicit DyadicInterval(Dyadic dyadic) : Dyadic(dyadic), Interval() {}
    DyadicInterval(Dyadic dyadic, IntervalType itype)
        : Dyadic(dyadic),
          Interval(itype)
    {}
    DyadicInterval(multiplier_t k, power_t n, IntervalType itype)
        : Dyadic(k, n),
          Interval(itype)
    {
        RPY_DBG_ASSERT(
                itype == IntervalType::Clopen || itype == IntervalType::Opencl
        );
    }
    DyadicInterval(
            Dyadic dyadic,
            power_t resolution,
            IntervalType itype = IntervalType::Clopen
    );
    DyadicInterval(
            param_t val,
            power_t resolution,
            IntervalType itype = IntervalType::Clopen
    );

    RPY_NO_DISCARD multiplier_t unit() const noexcept
    {
        return (m_interval_type == IntervalType::Clopen) ? 1 : -1;
    }

    using Dyadic::operator++;
    using Dyadic::operator--;
    using Dyadic::multiplier;
    using Dyadic::power;

    /**
     * @brief Returns the included end of the DyadicInterval.
     *
     * This method returns the dyadic number representing the included end of
     * the DyadicInterval. The included end is calculated based on the interval
     * type. If the interval type is Clopen, the included end is obtained using
     * `dincluded_end()`. Otherwise, it is obtained using `dexcluded_end()`.
     *
     * @return The dyadic number representing the included end of the
     * DyadicInterval.
     */
    RPY_NO_DISCARD Dyadic dincluded_end() const;
    /**
     * @brief Returns the excluded end of the DyadicInterval.
     *
     * This method returns the dyadic number representing the excluded end of
     * the DyadicInterval. The excluded end is calculated based on the interval
     * type. If the interval type is Clopen, the excluded end is obtained using
     * `dexcluded_end()`. Otherwise, it is obtained using `dincluded_end()`.
     *
     * @return The dyadic number representing the excluded end of the
     * DyadicInterval.
     */
    RPY_NO_DISCARD Dyadic dexcluded_end() const;
    /**
     * @brief Returns the infimum of the DyadicInterval as a dyadic number.
     *
     * This method returns the dyadic number representing the dinf of the
     * DyadicInterval. The dinf is calculated based on the interval type. If the
     * interval type is Clopen, the dinf is obtained using `dincluded_end()`.
     * Otherwise, it is obtained using `dexcluded_end()`.
     *
     * @return The dyadic number representing the dinf of the DyadicInterval.
     */
    RPY_NO_DISCARD Dyadic dinf() const;
    /**b
     * @brief Returns the supremum of the DyadicInterval as a dyadic number.
     *
     * This method calculates the supremum of the DyadicInterval based on
     * the interval type. If the interval type is Clopen, the excluded end is
     * obtained using `dexcluded_end()`. Otherwise, it is obtained using
     * `dincluded_end()`.
     *
     * @return The dyadic number representing the excluded end of the
     * DyadicInterval.
     */
    RPY_NO_DISCARD Dyadic dsup() const;

    /**
     * @brief Get the infimum of the dyadic interval.
     *
     * This method returns the lower bound of the dyadic interval.
     *
     * @return The infimum of the dyadic interval.
     */
    RPY_NO_DISCARD param_t inf() const override;
    /**
     * @brief Get the supremum of the dyadic interval.
     *
     * This function returns the supremum (i.e., the upper bound) of the dyadic
     * interval.
     *
     * @return The supremum of the dyadic interval.
     */
    RPY_NO_DISCARD param_t sup() const override;
    /**
     * @brief Returns the included end of the DyadicInterval.
     *
     * This method returns the dyadic number representing the included end of
     * the DyadicInterval. The included end is calculated based on the interval
     * type. If the interval type is Clopen, the included end is obtained using
     * `dincluded_end()`. Otherwise, it is obtained using `dexcluded_end()`.
     *
     * @return The param_t representing the included end of the
     * DyadicInterval.
     */
    RPY_NO_DISCARD param_t included_end() const override;
    /**
     * @brief Returns the excluded end of the DyadicInterval.
     *
     * This method returns the dyadic number representing the excluded end of
     * the DyadicInterval. The excluded end is calculated based on the interval
     * type. If the interval type is Clopen, the excluded end is obtained using
     * `dexcluded_end()`. Otherwise, it is obtained using `dincluded_end()`.
     *
     * @return The param_t representing the excluded end of the
     * DyadicInterval.
     */
    RPY_NO_DISCARD param_t excluded_end() const override;

    /**
     * @brief Shrinks the current DyadicInterval object to the specified power,
     * and returns the modified object.
     *
     * This method takes a power argument and shrinks the current DyadicInterval
     * object to the specified power. The resulting DyadicInterval object will
     * have the specified power as the end point, and the same start point and
     * interval type as the original object. The original object remains
     * unchanged.
     *
     * @param arg The power to shrink the DyadicInterval object to.
     *
     * @return A new DyadicInterval object that represents the current object
     * shrunk to the specified power.
     */
    RPY_NO_DISCARD DyadicInterval
    shrink_to_contained_end(power_t arg = 1) const;
    /**
     * @brief Shrink the DyadicInterval to the omitted end.
     *
     * This method shrinks the DyadicInterval to the omitted end, which means it
     * removes the rightmost interval from the DyadicInterval.
     *
     * @return A new DyadicInterval that is the result of shrinking the current
     * DyadicInterval to the omitted end.
     *
     * @see shrink_to_contained_end
     * @see flip_interval
     */
    RPY_NO_DISCARD DyadicInterval shrink_to_omitted_end() const;
    /**
     * @brief Shrinks the right end of the dyadic interval.
     *
     * This method shrinks the right end of the dyadic interval based on its
     * type. If the interval type is Opencl, it calls the
     * shrink_to_contained_end() method to perform the shrinkage. If the
     * interval type is not Opencl, it calls the shrink_to_omitted_end() method.
     *
     * @return A reference to the modified dyadic interval.
     */
    DyadicInterval& shrink_interval_right();
    /**
     * @brief Shrinks the left end of the dyadic interval by a specified power
     * of 2.
     *
     * @param arg The power of 2 by which the left end should be shrunk.
     *
     * @return A reference to the modified DyadicInterval object.
     *
     * @pre The value of arg must be greater than or equal to 0.
     *
     * This method shrinks the left end of the dyadic interval by dividing the
     * width by 2 a specified number of times (specified by arg).
     *
     * If the interval is of type Clopen, the left end is shrunk by dividing the
     * width and adjusting the start point to the next dyadic number contained
     * within the interval.
     *
     * If the interval is of type Omitted, the left end is shrunk by dividing
     * the width and adjusting the start point to the next dyadic number not
     * included in the interval.
     *
     * @note This method modifies the calling object.
     */
    DyadicInterval& shrink_interval_left(power_t arg = 1);
    /**
     * @brief Expands the dyadic interval.
     *
     * This method expands the current dyadic interval by subtracting the given
     * power from the interval's power value.
     *
     * @param arg The power value to subtract from the interval's power.
     *
     * @return A reference to the expanded DyadicInterval object.
     *
     * @see DyadicInterval
     */
    DyadicInterval& expand_interval(power_t arg = 1);

    /**
     * @brief Check if this dyadic interval contains another dyadic interval.
     *
     * This method checks if this dyadic interval contains the given dyadic
     * interval. The containment condition requires that both intervals have the
     * same interval type, and the other interval's power is not greater than
     * this interval's power. If these conditions are satisfied, the method
     * calculates the aligned value of the other interval and checks if it is
     * equal to the shifted value of this interval.
     *
     * @param other The dyadic interval to check for containment.
     * @return True if this dyadic interval contains the other dyadic interval,
     * false otherwise.
     */
    RPY_NO_DISCARD bool contains_dyadic(const DyadicInterval& other) const;
    /**
     * @brief Check if the DyadicInterval object is aligned.
     *
     * This method checks if the DyadicInterval object is aligned. It does so by
     * creating a parent DyadicInterval object, with the dyadic value and power
     * decremented by 1. Then it compares the current object with the parent
     * object to determine if they are equal. If they are equal, it means the
     * object is aligned.
     *
     * @return True if the DyadicInterval object is aligned, false otherwise.
     *
     * @see operator==()
     */
    RPY_NO_DISCARD bool aligned() const;

    /**
     * @brief Flips the dyadic interval.
     *
     * This method flips the dyadic interval by changing the value of the
     * multiplier. If the current multiplier is even, it is increased by the
     * unit value. If the current multiplier is odd, it is decreased by the unit
     * value.
     *
     * @return Reference to the modified dyadic interval.
     */
    DyadicInterval& flip_interval();

    /**
     * @brief Shifts the current DyadicInterval forward by a specified
     * multiplier.
     *
     * This method creates a new DyadicInterval object that is shifted forward
     * by the specified multiplier. The current DyadicInterval object is not
     * modified.
     *
     * @param arg The multiplier used to shift the DyadicInterval forward.
     *
     * @return A new DyadicInterval object that is shifted forward by the
     * specified multiplier.
     *
     * @see DyadicInterval
     * @see Dyadic
     * @see Interval
     */
    RPY_NO_DISCARD DyadicInterval shift_forward(multiplier_t arg = 1) const;
    /**
     * @brief Shifts the DyadicInterval back by a given multiplier.
     *
     * This method shifts the DyadicInterval back by multiplying the multiplier
     * value by the unit of the DyadicInterval. It creates a copy of the
     * DyadicInterval, performs the shift, and returns the shifted
     * DyadicInterval.
     *
     * @param arg The multiplier value to shift the DyadicInterval.
     * @return The shifted DyadicInterval.
     */
    RPY_NO_DISCARD DyadicInterval shift_back(multiplier_t arg = 1) const;

    /**
     * @fn DyadicInterval::advance() noexcept
     * @brief Advances the DyadicInterval by one unit.
     *
     * This member function modifies the state of the DyadicInterval object by
     * advancing its multiplier by one unit.
     *
     * @return Reference to the modified DyadicInterval object.
     */
    DyadicInterval& advance() noexcept;

    RPY_SERIAL_SERIALIZE_FN();
};

ROUGHPY_INTERVALS_EXPORT
std::ostream& operator<<(std::ostream& os, const DyadicInterval& di);

/**
 * @brief Converts a given Interval to a vector of DyadicIntervals.
 *
 * This function takes an input Interval and converts it to a vector of
 * DyadicIntervals with a given tolerance and IntervalType.
 *
 * @param interval The input Interval to be converted.
 * @param tol The tolerance value used for converting the Interval to
 * DyadicIntervals.
 * @param itype The IntervalType used for determining the structure of the
 * converted DyadicIntervals.
 * @return A vector of DyadicIntervals representing the converted Interval.
 */
ROUGHPY_INTERVALS_EXPORT
containers::Vec<DyadicInterval> to_dyadic_intervals(
        const Interval& interval,
        typename Dyadic::power_t tol,
        IntervalType itype = IntervalType::Clopen
);

ROUGHPY_INTERVALS_EXPORT
bool operator<(const DyadicInterval& lhs, const DyadicInterval& rhs) noexcept;

#ifdef RPY_COMPILING_INTERVALS
RPY_SERIAL_EXTERN_SERIALIZE_CLS_BUILD(DyadicInterval)
#else
RPY_SERIAL_EXTERN_SERIALIZE_CLS_IMP(DyadicInterval)
#endif

RPY_SERIAL_SERIALIZE_FN_IMPL(DyadicInterval)
{
    RPY_SERIAL_SERIALIZE_BASE(Interval);
    RPY_SERIAL_SERIALIZE_BASE(Dyadic);
}

}// namespace intervals
}// namespace rpy

RPY_SERIAL_SPECIALIZE_TYPES(
        ::rpy::intervals::DyadicInterval,
        ::rpy::serial::specialization::member_serialize
);

#endif// ROUGHPY_INTERVALS_DYADIC_INTERVAL_H_
