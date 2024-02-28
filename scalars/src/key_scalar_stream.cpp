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

#include "key_scalar_stream.h"

#include "key_scalar_array.h"
#include "scalar_stream.h"

using namespace rpy;
using namespace rpy::scalars;
KeyScalarStream::KeyScalarStream() : ScalarStream(), m_key_stream() {}
KeyScalarStream::KeyScalarStream(const ScalarType* type)
    : ScalarStream(type),
      m_key_stream()
{}
KeyScalarStream::KeyScalarStream(const KeyScalarStream& other)
    : ScalarStream(other),
      m_key_stream(other.m_key_stream)
{}
KeyScalarStream::KeyScalarStream(KeyScalarStream&& other) noexcept
    : ScalarStream(static_cast<ScalarStream&&>(other)),
      m_key_stream(std::move(other.m_key_stream))
{}
KeyScalarStream& KeyScalarStream::operator=(const KeyScalarStream& other)
{
    if (&other != this) {
        this->~KeyScalarStream();
        ScalarStream::operator=(static_cast<const ScalarStream&>(other));
        m_key_stream = other.m_key_stream;
    }
    return *this;
}
KeyScalarStream& KeyScalarStream::operator=(KeyScalarStream&& other) noexcept
{
    if (&other != this) {
        this->~KeyScalarStream();
        ScalarStream::operator=(static_cast<ScalarStream&&>(other));
        m_key_stream = std::move(other.m_key_stream);
    }
    return *this;
}
KeyScalarArray KeyScalarStream::operator[](dimn_t row) const noexcept {
    RPY_CHECK(row < m_stream.size());

    return {
        ScalarStream::operator[](row),
        m_key_stream.empty() ? nullptr : m_key_stream[row]
    };
}
void KeyScalarStream::reserve_size(dimn_t num_rows)
{
    ScalarStream::reserve_size(num_rows);
    m_key_stream.reserve(num_rows);
}
void KeyScalarStream::push_back(
        const ScalarArray& scalar_data,
        const key_type* key_ptr
)
{
    if (key_ptr != nullptr) {
        if (m_key_stream.empty()) {
            m_key_stream.resize(ScalarStream::row_count(), nullptr);
        }
        m_key_stream.push_back(key_ptr);
    }
    ScalarStream::push_back(scalar_data);
}
void KeyScalarStream::push_back(
        ScalarArray&& scalar_data,
        const key_type* key_ptr
)
{
    if (key_ptr != nullptr) {
        if (m_key_stream.empty()) {
            m_key_stream.resize(ScalarStream::row_count(), nullptr);
        }
        m_key_stream.push_back(key_ptr);
    }
    ScalarStream::push_back(std::move(scalar_data));
}
