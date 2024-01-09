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

#ifndef ROUGHPY_SCALARS_KEY_SCALAR_ARRAY_H_
#define ROUGHPY_SCALARS_KEY_SCALAR_ARRAY_H_

#include "scalar_array.h"
#include "scalars_fwd.h"

#include <roughpy/core/types.h>
#include <roughpy/platform/serialization.h>

#include <cereal/types/vector.hpp>
#include <algorithm>

// RPY_WARNING_PUSH
// RPY_CLANG_DISABLE_WARNING(HidingNonVirtualFunction)

namespace rpy {
namespace scalars {

class ROUGHPY_SCALARS_EXPORT KeyScalarArray : public ScalarArray
{
    const key_type* p_keys = nullptr;
    bool m_owns_keys = false;

public:
    KeyScalarArray() = default;
    ~KeyScalarArray();

    KeyScalarArray(const KeyScalarArray& other);
    KeyScalarArray(KeyScalarArray&& other) noexcept;

    explicit KeyScalarArray(ScalarArray&& sa) noexcept;
    KeyScalarArray(ScalarArray base, const key_type* keys);

    explicit KeyScalarArray(const ScalarType* type) noexcept;
    explicit KeyScalarArray(const ScalarType* type, dimn_t n) noexcept;
    KeyScalarArray(
            const ScalarType* type,
            const void* begin,
            dimn_t count
    ) noexcept;


    RPY_NO_DISCARD KeyScalarArray copy_or_move() &&;

    KeyScalarArray& operator=(const KeyScalarArray& other);
    KeyScalarArray& operator=(const ScalarArray& other);
    KeyScalarArray& operator=(KeyScalarArray&& other) noexcept;
    KeyScalarArray& operator=(ScalarArray&& other) noexcept;

    RPY_NO_DISCARD const key_type* keys() const noexcept { return p_keys; }
    RPY_NO_DISCARD key_type* keys();
    RPY_NO_DISCARD bool has_keys() const noexcept { return p_keys != nullptr; }

    void allocate_scalars(idimn_t count = -1);
    void allocate_keys(idimn_t count = -1);


    RPY_SERIAL_LOAD_FN();
    RPY_SERIAL_SAVE_FN();

};

RPY_SERIAL_LOAD_FN_IMPL(KeyScalarArray)
{
    RPY_SERIAL_SERIALIZE_BASE(ScalarArray);
    bool hkeys = false;
    RPY_SERIAL_SERIALIZE_NVP("has_keys", hkeys);
    if (hkeys) {
        std::vector<key_type> tmp_keys;
        RPY_SERIAL_SERIALIZE_NVP("keys", tmp_keys);

        RPY_CHECK(tmp_keys.size() == size());
        allocate_keys();
        std::memcpy(keys(), tmp_keys.data(), size() * sizeof(key_type));
        // std::copy(tmp_keys.begin(), tmp_keys.end(), keys());
    }
}
RPY_SERIAL_SAVE_FN_IMPL(KeyScalarArray)
{
    RPY_SERIAL_SERIALIZE_BASE(ScalarArray);
    bool hkeys = has_keys();
    RPY_SERIAL_SERIALIZE_NVP("has_keys", hkeys);
    if (hkeys) {
        std::vector<key_type> tmp_keys(p_keys, p_keys + size());
        RPY_SERIAL_SERIALIZE_NVP("keys", tmp_keys);
    }
}


}// namespace scalars
}// namespace rpy


// RPY_WARNING_POP
#endif// ROUGHPY_SCALARS_KEY_SCALAR_ARRAY_H_
