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

#ifndef ROUGHPY_SCALARS_SCALARS_FWD_H_
#define ROUGHPY_SCALARS_SCALARS_FWD_H_

#include <roughpy/platform/devices/core.h>
#include <roughpy/platform/devices/macros.h>
#include <roughpy/platform/errors.h>

#include "roughpy_scalars_export.h"

namespace rpy {
namespace scalars {

using ScalarTypeCode = devices::TypeCode;
using BasicScalarInfo = devices::TypeInfo;

using seed_int_t = uint64_t;

// Forward declarations
class ROUGHPY_SCALARS_EXPORT ScalarType;
class ROUGHPY_SCALARS_EXPORT ScalarInterface;
class ROUGHPY_SCALARS_EXPORT Scalar;
class ROUGHPY_SCALARS_EXPORT ScalarArray;
class ScalarArrayView;
class ROUGHPY_SCALARS_EXPORT KeyScalarArray;
class ROUGHPY_SCALARS_EXPORT ScalarStream;
class ROUGHPY_SCALARS_EXPORT KeyScalarStream;
class ROUGHPY_SCALARS_EXPORT RandomGenerator;

namespace dtl {

template <typename T>
struct ScalarTypeOfImpl {
    static optional<const ScalarType*> get() noexcept;
};

template <typename T>
inline optional<const ScalarType*> ScalarTypeOfImpl<T>::get() noexcept
{
    return {};
}

template <>
ROUGHPY_SCALARS_EXPORT optional<const ScalarType*>
ScalarTypeOfImpl<float>::get() noexcept;

template <>
ROUGHPY_SCALARS_EXPORT optional<const ScalarType*>
ScalarTypeOfImpl<double>::get() noexcept;

template <>
ROUGHPY_SCALARS_EXPORT optional<const ScalarType*>
ScalarTypeOfImpl<devices::rational_scalar_type>::get() noexcept;

template <>
ROUGHPY_SCALARS_EXPORT optional<const ScalarType*>
ScalarTypeOfImpl<devices::rational_poly_scalar>::get() noexcept;

template <>
ROUGHPY_SCALARS_EXPORT optional<const ScalarType*>
ScalarTypeOfImpl<devices::half>::get() noexcept;

template <>
ROUGHPY_SCALARS_EXPORT optional<const ScalarType*>
ScalarTypeOfImpl<devices::bfloat16>::get() noexcept;

}// namespace dtl

template <typename T>
RPY_NO_DISCARD optional<const ScalarType*> scalar_type_of()
{
    return dtl::ScalarTypeOfImpl<T>::get();
}

RPY_NO_DISCARD ROUGHPY_SCALARS_EXPORT optional<const ScalarType*>
scalar_type_of(devices::TypeInfo info);

RPY_NO_DISCARD ROUGHPY_SCALARS_EXPORT optional<const ScalarType*>
scalar_type_of(devices::TypeInfo info, const devices::Device& device);

template <typename T>
RPY_NO_DISCARD T scalar_cast(const Scalar& value);

RPY_NO_DISCARD ROUGHPY_SCALARS_EXPORT optional<const ScalarType*>
get_type(string_view id);

inline constexpr int min_scalar_type_alignment = 16;

#define RPY_SCALAR_TYPE_ALIGNMENT alignas(min_scalar_type_alignment)

/**
 * @brief Determines the type promotion when performing an
 * operation with two types.
 *
 * This function takes two types and returns the result of the
 * type promotion according to the following rules:
 * - If both types are the same, the result is the same type.
 * - If one of the types is an integer and the other is a
 * floating-point, the result is a floating-point type.
 * - If one of the types is a long integer and the other is a
 * regular integer, the result is a long integer type.
 * - If one of the types is a long integer and the other is a
 * floating-point, the result is a floating-point type.
 * - If one of the types is a double precision floating-point and
 * the other is a regular floating-point, the result is a double
 * precision floating-point type.
 *
 * @param left The type of the left operand.
 * @param right The type of the right operand.
 * @return The resulting type after performing promotion on
 * `left` and `right`.
 */

RPY_NO_DISCARD ROUGHPY_SCALARS_EXPORT devices::TypeInfo
compute_type_promotion(devices::TypeInfo left, devices::TypeInfo right);

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SCALARS_FWD_H_
