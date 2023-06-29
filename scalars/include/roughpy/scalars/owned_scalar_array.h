// Copyright (c) 2023 RoughPy Developers. All rights reserved.
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

#ifndef ROUGHPY_SCALARS_OWNED_SCALAR_ARRAY_H_
#define ROUGHPY_SCALARS_OWNED_SCALAR_ARRAY_H_

#include "scalars_fwd.h"

#include "scalar_array.h"

#include <roughpy/platform/serialization.h>

namespace rpy { namespace scalars {

class RPY_EXPORT OwnedScalarArray : public ScalarArray {
public:
    OwnedScalarArray() = default;

    OwnedScalarArray(const OwnedScalarArray &other);
    OwnedScalarArray(OwnedScalarArray &&other) noexcept;

    explicit OwnedScalarArray(const ScalarType *type);
    OwnedScalarArray(const ScalarType *type, dimn_t size);
    OwnedScalarArray(const Scalar &value, dimn_t count);
    explicit OwnedScalarArray(const ScalarArray &other);

    explicit OwnedScalarArray(const ScalarType *type, const void *data, dimn_t count);

    OwnedScalarArray &operator=(const ScalarArray &other);
    OwnedScalarArray &operator=(OwnedScalarArray &&other) noexcept;

    ~OwnedScalarArray();

    RPY_SERIAL_SERIALIZE_FN();


};

RPY_SERIAL_SERIALIZE_FN_IMPL(OwnedScalarArray) {
    RPY_SERIAL_SERIALIZE_BASE(ScalarArray);
}


}}

#endif // ROUGHPY_SCALARS_OWNED_SCALAR_ARRAY_H_
