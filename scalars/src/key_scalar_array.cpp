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

#include "key_scalar_array.h"
#include "scalar_type.h"

#include <algorithm>

using namespace rpy;
using namespace rpy::scalars;

KeyScalarArray::~KeyScalarArray()
{
    if (p_keys != nullptr && m_owns_keys) { delete[] p_keys; }
    p_keys = nullptr;
    m_owns_keys = false;
}
KeyScalarArray::KeyScalarArray(const KeyScalarArray& other)
    : ScalarArray(other),
      p_keys()
{
    if (other.p_keys != nullptr && other.m_owns_keys) {
        m_owns_keys = true;
        allocate_keys();
        std::copy_n(other.p_keys, other.size(), const_cast<key_type*>(p_keys));
    }
}
KeyScalarArray::KeyScalarArray(KeyScalarArray&& other) noexcept
    : ScalarArray(std::move(other)), p_keys(other.p_keys), m_owns_keys(other.m_owns_keys)
{
    other.p_keys = nullptr;
    other.m_owns_keys = false;
}
KeyScalarArray::KeyScalarArray(ScalarArray&& sa) noexcept
    : ScalarArray(std::move(sa)), m_owns_keys(false)
{}
KeyScalarArray::KeyScalarArray(ScalarArray base, const key_type* keys)
    : ScalarArray(std::move(base)), p_keys(keys), m_owns_keys(false)
{}
KeyScalarArray::KeyScalarArray(const ScalarType* type) noexcept
    : ScalarArray(type), m_owns_keys(false)
{}
KeyScalarArray::KeyScalarArray(const ScalarType* type, dimn_t n) noexcept
    : ScalarArray(type, n), m_owns_keys(false)
{}
KeyScalarArray::KeyScalarArray(
        const ScalarType* type,
        const void* begin,
        dimn_t count
) noexcept
    : ScalarArray(type, begin, count), m_owns_keys(false)
{}
KeyScalarArray KeyScalarArray::copy_or_move() && {
    if (m_owns_keys) {
        return std::move(*this);
    }

    KeyScalarArray result(static_cast<ScalarArray&&>(*this).copy_or_clone());
    result.allocate_keys();
    std::copy_n(p_keys, size(), result.keys());

    return result;
}
KeyScalarArray& KeyScalarArray::operator=(const KeyScalarArray& other)
{
    if (&other != this) {
        this->~KeyScalarArray();
        ScalarArray::operator=(other);
        if (other.m_owns_keys) {
            allocate_keys();
            std::copy_n(other.p_keys, other.size(), keys());
            m_owns_keys = true;
        } else {
            p_keys = other.p_keys;
            m_owns_keys = false;
        }

    }
    return *this;
}

KeyScalarArray& KeyScalarArray::operator=(const ScalarArray& other)
{
    if (&other != this) {
        this->~KeyScalarArray();
        ScalarArray::operator=(other);
    }
    return *this;
}
KeyScalarArray& KeyScalarArray::operator=(KeyScalarArray&& other) noexcept
{
    if (&other != this) {
        this->~KeyScalarArray();

        ScalarArray::operator=(std::move(other));
        p_keys = other.p_keys;
        other.p_keys = nullptr;
        m_owns_keys = other.m_owns_keys;

    }
    return *this;
}
KeyScalarArray& KeyScalarArray::operator=(ScalarArray&& other) noexcept
{
    if (&other != this) {
        this->~KeyScalarArray();
        ScalarArray::operator=(std::move(other));
    }
    return *this;
}
key_type* KeyScalarArray::keys() {
    if (m_owns_keys) {
        return const_cast<key_type *>(p_keys);
    }
    return nullptr;
}
void KeyScalarArray::allocate_scalars(idimn_t count) {
    auto type = this->type();
    if (count >= 0 && type) {
        *this = (*type)->allocate(static_cast<dimn_t>(count));
    }
}
void KeyScalarArray::allocate_keys(idimn_t count) {
    if (count != 0) {
        if (p_keys != nullptr) { delete[] p_keys; }

        if (count == -1) {
            count = static_cast<idimn_t>(size());
        }

        RPY_CHECK(count == static_cast<idimn_t>(size()));

        p_keys = new key_type[count] {};
        m_owns_keys = true;
    }

}



#define RPY_SERIAL_IMPL_CLASSNAME KeyScalarArray
#define RPY_SERIAL_DO_SPLIT
#include <roughpy/platform/serialization_instantiations.inl>
