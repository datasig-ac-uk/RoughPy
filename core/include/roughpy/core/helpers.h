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

#ifndef ROUGHPY_CORE_POINTER_HELPERS_H_
#define ROUGHPY_CORE_POINTER_HELPERS_H_

#include "macros.h"
#include "traits.h"
#include "types.h"

#include <bitset>
#include <climits>
#include <cstring>
#include <iterator>
#include <vector>

namespace rpy {

/**
 * @brief Cast the bit value from a value of type From to a value
 * of type To.
 *
 * The implementation here is pretty similar to the example from
 * https://en.cppreference.com/w/cpp/numeric/bit_cast
 * if we have to define our own version using memcpy.
 */
#if defined(__cpp_lib_bit_cast) && __cpp_Lib_bit_cast >= 201806L
using std::bit_cast;
#else
template <typename To, typename From>
RPY_NO_DISCARD enable_if_t<
        sizeof(To) == sizeof(From)
                && is_trivially_copyable_v<From> && is_trivially_copyable_v<To>
                && is_default_constructible_v<To>,
        To>
bit_cast(From from)
{
    To to;
    memcpy(static_cast<void*>(std::addressof(to)),
           static_cast<const void*>(std::addressof(from)),
           sizeof(To));
    return to;
}
#endif

template <typename T>
RPY_NO_DISCARD inline enable_if_t<is_integral_v<T>, size_t> count_bits(T val
) noexcept
{
    // There might be optimisations using builtin popcount functions
    return std::bitset<CHAR_BIT * sizeof(T)>(val).count();
}

RPY_NO_DISCARD constexpr size_t static_log2p1(size_t value) noexcept
{
    return (value == 0) ? 0 : 1 + static_log2p1(value >> 1);
}

template <typename I, typename J>
RPY_NO_DISCARD constexpr enable_if_t<is_integral_v<I>, I>
round_up_divide(I value, J divisor) noexcept
{
    return (value + static_cast<I>(divisor) - 1) / static_cast<I>(divisor);
}

template <typename I>
RPY_NO_DISCARD constexpr enable_if_t<is_integral_v<I>, I>
next_power_2(I value, I start = I(1)) noexcept
{
    if (value == 0) { return 0; }
    if (is_signed_v<I> && value < 0) { return -next_power_2(-value); }
    return (start >= value) ? start : next_power_2(value, I(start << 1));
}

template <unsigned Base, typename I>
/**
 * @brief Calculates the logarithm base @p Base of an integral value @p arg
 * recursively.
 *
 * This function uses a constexpr implementation to recursively calculate the
 * logarithm base @p Base of an integral value @p arg. It returns the number of
 * times @p Base must be multiplied by itself to reach or exceed the value @p
 * arg.
 *
 * @tparam I The type of the integral value.
 * @param arg The integral value for which to calculate the logarithm.
 * @return The logarithm base @p Base of the integral value @p arg.
 */
RPY_NO_DISCARD constexpr enable_if_t<is_integral_v<I>, I> const_log(I arg
) noexcept
{
    return arg >= static_cast<I>(Base)
            ? 1 + const_log<Base>(static_cast<I>(arg / Base))
            : 0;
}

template <typename I>
RPY_NO_DISCARD enable_if_t<is_integral_v<I>, bool> is_odd(I arg) noexcept
{
    return (arg % 2) == 1;
}

template <typename I>
RPY_NO_DISCARD enable_if_t<is_integral_v<I>, bool> is_even(I arg) noexcept
{
    return (arg % 2) == 0;
}

template <typename I, typename E>
RPY_NO_DISCARD constexpr enable_if_t<is_integral_v<E>, I>
const_power(I arg, E exponent) noexcept
{
    if (exponent == 0) { return 1; }
    if (exponent == 1) { return arg; }

    const auto half_power = const_power(arg, exponent / 2);

    if (is_odd(exponent)) { return arg * half_power * half_power; }

    return half_power * half_power;
}

/**
 * Calculates the quotient and remainder of dividing a given number by a given
 * divisor.
 *
 * This function calculates the quotient and remainder of the division of the
 * given numerator (`num`) and divisor (`divisor`). The result is returned as a
 * pair, where the first element represents the quotient and the second element
 * represents the remainder.
 *
 * @param num The number to be divided.
 * @param divisor The divisor.
 * @return A pair containing the quotient and remainder of the division.
 *
 * @note The types `I` and `J` must satisfy the requirements:
 *       - `I` must be an integral type.
 *       - `J` must be an integral type.
 *       - `J` must be convertible to `I`.
 *
 * @note The function is constexpr and noexcept, ensuring it can be used in
 *       compile-time evaluation and it does not throw any exceptions.
 */
template <typename I, typename J>
constexpr enable_if_t<
        is_integral_v<I> && is_integral_v<J> && is_convertible_v<J, I>,
        pair<I, J>>
remquo(I num, J divisor) noexcept
{
    const auto div = static_cast<I>(divisor);
    return {num / div, static_cast<J>(num % div)};
}
}// namespace rpy

#endif// ROUGHPY_CORE_POINTER_HELPERS_H_
