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

#ifndef ROUGHPY_SCALARS_TRAITS_H_
#define ROUGHPY_SCALARS_TRAITS_H_

#include <roughpy/platform/devices/core.h>
#include <roughpy/core/helpers.h>

namespace rpy {
namespace scalars {
namespace traits {

constexpr dimn_t size_of(const devices::TypeInfo& info) noexcept
{
    return static_cast<dimn_t>(info.bytes);
}

constexpr dimn_t align_of(const devices::TypeInfo& info) noexcept {
    return dimn_t(1) << static_log2p1(info.bytes);
}

constexpr bool is_floating_point(const devices::TypeInfo& info) noexcept
{
    return info.code == devices::TypeCode::Float
            || info.code == devices::TypeCode::BFloat;
}

constexpr bool is_integral(const devices::TypeInfo& info) noexcept
{
    return info.code == devices::TypeCode::Int
            || info.code == devices::TypeCode::UInt;
}

constexpr bool is_arithmetic(const devices::TypeInfo& info) noexcept
{
    return is_integral(info) || is_floating_point(info);
}

constexpr bool is_fundamental(const devices::TypeInfo& info) noexcept {
    return is_arithmetic(info) || info.code == devices::TypeCode::Bool;
}

constexpr bool is_signed(const devices::TypeInfo& info) noexcept
{
    switch (info.code) {
        case devices::TypeCode::UInt:
        case devices::TypeCode::OpaqueHandle:
        case devices::TypeCode::Complex:
        case devices::TypeCode::Bool:
        case devices::TypeCode::ArbitraryPrecisionUInt:
        case devices::TypeCode::ArbitraryPrecisionComplex:
        case devices::TypeCode::APRationalPolynomial: return false;
        case devices::TypeCode::Int:
        case devices::TypeCode::Float:
        case devices::TypeCode::BFloat:
        case devices::TypeCode::Rational:
        case devices::TypeCode::ArbitraryPrecisionInt:
        case devices::TypeCode::ArbitraryPrecisionFloat:
        case devices::TypeCode::ArbitraryPrecisionRational: return true;
    }
    RPY_UNREACHABLE_RETURN(false);
}

constexpr bool is_unsigned(const devices::TypeInfo& info) noexcept
{
    return !is_signed(info);
}


constexpr devices::TypeInfo rational_type_of(const devices::TypeInfo& info) noexcept
{
    switch (info.code) {
        case devices::TypeCode::Int:
        case devices::TypeCode::UInt:
        case devices::TypeCode::Float:
        case devices::TypeCode::OpaqueHandle:
        case devices::TypeCode::BFloat:
        case devices::TypeCode::Complex:
        case devices::TypeCode::Bool:
        case devices::TypeCode::Rational:
        case devices::TypeCode::ArbitraryPrecisionInt:
        case devices::TypeCode::ArbitraryPrecisionUInt:
        case devices::TypeCode::ArbitraryPrecisionFloat:
        case devices::TypeCode::ArbitraryPrecisionComplex:
        case devices::TypeCode::ArbitraryPrecisionRational:
            return info;
        case devices::TypeCode::APRationalPolynomial:
            return devices::type_info<devices::rational_poly_scalar>();
    }

    return info;
}


}// namespace traits
}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_TRAITS_H_
