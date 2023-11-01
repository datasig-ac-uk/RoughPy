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

#ifndef ROUGHPY_SCALARS_SCALARS_FWD_H_
#define ROUGHPY_SCALARS_SCALARS_FWD_H_

#include <roughpy/device/macros.h>
#include <roughpy/device/core.h>

namespace rpy { namespace scalars {

using ScalarTypeCode = devices::TypeCode;
using BasicScalarInfo = devices::TypeInfo;

// Forward declarations
class ScalarType;
class ScalarInterface;
class Scalar;
class ScalarArray;
class ScalarArrayView;
class KeyScalarArray;
class ScalarStream;
class KeyScalarStream;
class RandomGenerator;


namespace dtl {

template <typename T>
struct ScalarTypeOfImpl {
    static const ScalarType* get() noexcept {
        static_assert(false, "no scalar type exists for this type");
    }
};

template <>
RPY_EXPORT const ScalarType* ScalarTypeOfImpl<float>::get() noexcept;

}

template <typename T>
RPY_NO_DISCARD const ScalarType* scalar_type_of() {
    return dtl::ScalarTypeOfImpl<T>::get();
}

RPY_NO_DISCARD const ScalarType* scalar_type_of(devices::TypeInfo info);


template <typename T>
RPY_NO_DISCARD T scalar_cast(const Scalar& value);


inline constexpr int min_scalar_type_alignment = 16;



}}

#endif // ROUGHPY_SCALARS_SCALARS_FWD_H_
