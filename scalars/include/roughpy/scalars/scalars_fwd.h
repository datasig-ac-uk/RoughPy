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

#include <roughpy/platform/errors.h>

#include "roughpy_scalars_export.h"

#include <roughpy/devices/core.h>

#include <roughpy/devices/type.h>
#include <roughpy/devices/value.h>

namespace rpy {
namespace scalars {
using ScalarTypeCode = devices::TypeCode;
using BasicScalarInfo = devices::TypeInfo;

using seed_int_t = uint64_t;

using devices::get_type;
using devices::Type;
using devices::TypePtr;

using Scalar = devices::Value;
using ScalarCRef = devices::ConstReference;
using ScalarRef = devices::Reference;

// Forward declarations
class ROUGHPY_SCALARS_EXPORT ScalarInterface;
class ROUGHPY_SCALARS_EXPORT ScalarArray;
class ROUGHPY_SCALARS_EXPORT ScalarStream;
class ROUGHPY_SCALARS_EXPORT RandomGenerator;

template <typename T, typename S>
RPY_NO_DISCARD
        enable_if_t<is_default_constructible_v<T> && devices::value_like<S>, T>
        scalar_cast(const S& value)
{
    const auto type = devices::get_type<T>();
    const auto val_type = value.type();
    T result{};
    if (type == val_type) {
        result = *value.template data<T>();
    } else {
        const auto& conversions = type->conversions(val_type);
        RPY_CHECK(conversions.convert != nullptr);
        conversions.convert(&result, value.data());
    }
    return result;
}

inline TypePtr rational_type_of(const TypePtr& type)
{
    RPY_DBG_ASSERT(type != nullptr);

    const auto* num_traits = type->num_traits();
    RPY_CHECK(num_traits != nullptr);

    return num_traits->rational_type;
}

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SCALARS_FWD_H_
