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

#ifndef ROUGHPY_SCALARS_TRAITS_H_
#define ROUGHPY_SCALARS_TRAITS_H_

#include <roughpy/devices/core.h>
#include <roughpy/core/helpers.h>

namespace rpy {
namespace scalars {
namespace traits {

/**
 * @brief Get the size of a TypeInfo object in bytes.
 *
 * This function returns the size of the TypeInfo object in bytes.
 *
 * @param info The TypeInfo object to get the size of.
 *
 * @return The size of the TypeInfo object in bytes.
 */
constexpr dimn_t size_of(const devices::TypeInfo& info) noexcept
{
    return static_cast<dimn_t>(info.bytes);
}

/**
 * @brief Get the alignment of a TypeInfo object.
 *
 * This constexpr function calculates and returns the alignment of a TypeInfo
 * object based on its bytes value. The alignment is calculated using the
 * formula 1 << static_log2p1(info.bytes).
 *
 * @param info       The TypeInfo object to get the alignment of.
 *
 * @return The alignment of the TypeInfo object.
 */
constexpr dimn_t align_of(const devices::TypeInfo& info) noexcept
{
    return dimn_t(1) << static_log2p1(info.bytes);
}

/**
 * @brief Check if the given TypeInfo object represents a floating point type.
 *
 * This function checks if the provided TypeInfo object represents a floating
 * point type. It returns true if the TypeInfo object's code is either
 * devices::TypeCode::Float or devices::TypeCode::BFloat, indicating a floating
 * point type. Otherwise, it returns false.
 *
 * @param info The TypeInfo object to check.
 *
 * @return true if the TypeInfo object represents a floating point type,
 * otherwise false.
 */
constexpr bool is_floating_point(const devices::TypeInfo& info) noexcept
{
    return info.code == devices::TypeCode::Float
            || info.code == devices::TypeCode::BFloat;
}

/**
 * @brief Check if a given TypeInfo object represents an integral type.
 *
 * This function checks if the given TypeInfo object represents an integral
 * type. It returns true if the type code of the TypeInfo object is either Int
 * or UInt, and false otherwise.
 *
 * @param info The TypeInfo object to check.
 *
 * @return True if the TypeInfo object represents an integral type, false
 * otherwise.
 */
constexpr bool is_integral(const devices::TypeInfo& info) noexcept
{
    return info.code == devices::TypeCode::Int
            || info.code == devices::TypeCode::UInt;
}

/**
 * @brief Check if a TypeInfo object represents an arithmetic type.
 *
 * This function checks if the given TypeInfo object represents an arithmetic
 * type. An arithmetic type is either an integral type or a floating-point type.
 *
 * @param info The TypeInfo object to check.
 *
 * @return A boolean value indicating if the TypeInfo object represents an
 * arithmetic type or not.
 */
constexpr bool is_arithmetic(const devices::TypeInfo& info) noexcept
{
    return is_integral(info) || is_floating_point(info);
}

/**
 * @brief Check if the given TypeInfo is a fundamental type.
 *
 * This function checks if the given TypeInfo object represents a fundamental
 * type. A fundamental type is either an arithmetic type or the TypeCode is
 * Bool.
 *
 * @param info The TypeInfo object to check.
 *
 * @return True if the TypeInfo object is a fundamental type, false otherwise.
 */
constexpr bool is_fundamental(const devices::TypeInfo& info) noexcept
{
    return is_arithmetic(info) || info.code == devices::TypeCode::Bool;
}

/**
 * @brief Check if a TypeInfo object represents a signed type.
 *
 * This function takes a TypeInfo object and checks if it represents a signed
 * type.
 *
 * @param info The TypeInfo object to check.
 *
 * @return true if the TypeInfo object represents a signed type, false
 * otherwise.
 */
constexpr bool is_signed(const devices::TypeInfo& info) noexcept
{
    switch (info.code) {
        case devices::TypeCode::Int:
        case devices::TypeCode::Float:
        case devices::TypeCode::BFloat:
        case devices::TypeCode::Rational:
        case devices::TypeCode::ArbitraryPrecisionInt:
        case devices::TypeCode::ArbitraryPrecisionFloat:
        case devices::TypeCode::ArbitraryPrecisionRational: return true;
        case devices::TypeCode::UInt:
        case devices::TypeCode::OpaqueHandle:
        case devices::TypeCode::Complex:
        case devices::TypeCode::Bool:
        case devices::TypeCode::ArbitraryPrecisionUInt:
        case devices::TypeCode::ArbitraryPrecisionComplex:
        case devices::TypeCode::APRationalPolynomial:
        case devices::TypeCode::KeyType: return false;
    }
    RPY_UNREACHABLE_RETURN(false);
}

/**
 * @brief Check if the TypeInfo object is unsigned.
 *
 * This function checks if the TypeInfo object represents an unsigned data type.
 *
 * @param info The TypeInfo object to check.
 *
 * @return true if the TypeInfo object represents an unsigned data type,
 * false otherwise.
 */
constexpr bool is_unsigned(const devices::TypeInfo& info) noexcept
{
    return !is_signed(info);
}

}// namespace traits
}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_TRAITS_H_
