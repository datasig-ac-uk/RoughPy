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

#ifndef ROUGHPY_SCALARS_KEY_SCALAR_ARRAY_H_
#define ROUGHPY_SCALARS_KEY_SCALAR_ARRAY_H_

#include "scalars_fwd.h"

#include "scalar_array.h"
#include "scalar_type.h"

namespace rpy {
namespace scalars {

class ROUGHPY_SCALARS_EXPORT KeyScalarArray : public ScalarArray {

    static constexpr uint32_t keys_owning_flag = 1 << subtype_flag_offset;

    const key_type *p_keys = nullptr;

    constexpr bool keys_owned() const noexcept { return (m_flags & keys_owning_flag) != 0; }

public:
    KeyScalarArray() = default;
    ~KeyScalarArray();

    KeyScalarArray(const KeyScalarArray &other);
    KeyScalarArray(KeyScalarArray &&other) noexcept;

    explicit KeyScalarArray(OwnedScalarArray &&sa) noexcept;
    KeyScalarArray(ScalarArray base, const key_type *keys);

    explicit KeyScalarArray(const ScalarType *type) noexcept;
    explicit KeyScalarArray(const ScalarType *type, dimn_t n) noexcept;
    KeyScalarArray(const ScalarType *type, const void *begin, dimn_t count) noexcept;

    explicit operator OwnedScalarArray() &&noexcept;

    KeyScalarArray copy_or_move() &&;

    KeyScalarArray &operator=(const ScalarArray &other) noexcept;
    KeyScalarArray &operator=(KeyScalarArray &&other) noexcept;
    KeyScalarArray &operator=(OwnedScalarArray &&other) noexcept;

    const key_type *keys() const noexcept { return p_keys; }
    key_type *keys();
    bool has_keys() const noexcept { return p_keys != nullptr; }

    void allocate_scalars(idimn_t count = -1);
    void allocate_keys(idimn_t count = -1);
};
}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_KEY_SCALAR_ARRAY_H_
