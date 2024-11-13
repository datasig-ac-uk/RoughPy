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

#include "scalar_stream.h"

#include "roughpy/core/check.h"  // for throw_exception, RPY_CHECK

#include "scalar.h"
#include "scalar_array.h"

using namespace rpy;
using namespace rpy::scalars;

ScalarStream::ScalarStream() : m_stream(),  p_type(nullptr) {}
ScalarStream::ScalarStream(const ScalarType* type)
    : m_stream(),
      p_type(type)
{}

ScalarStream::ScalarStream(const ScalarStream& other)
    : m_stream(other.m_stream),
      p_type(other.p_type)
{}
ScalarStream::ScalarStream(ScalarStream&& other) noexcept
    : m_stream(std::move(other.m_stream)),
      p_type(other.p_type)
{}

ScalarStream::ScalarStream(ScalarArray base, std::vector<dimn_t> shape)
{
    if (!base.is_null()) {
        auto tp = base.type();
        RPY_CHECK(tp);
        p_type = *tp;

        if (shape.empty()) {
            RPY_THROW(std::runtime_error, "strides cannot be empty");
        }

        dimn_t rows = shape[0];
        dimn_t cols = (shape.size() > 1) ? shape[1] : 1;


        m_stream.reserve(rows);

        for (dimn_t i = 0; i < rows; ++i) {
            m_stream.push_back(base[{i * cols, (i + 1) * cols}]);
        }
    }
}

ScalarStream& ScalarStream::operator=(const ScalarStream& other)
{
    if (&other != this) {
        this->~ScalarStream();
        m_stream = other.m_stream;
        p_type = other.p_type;
    }
    return *this;
}
ScalarStream& ScalarStream::operator=(ScalarStream&& other) noexcept
{
    if (&other != this) {
        this->~ScalarStream();
        m_stream = std::move(other.m_stream);
        p_type = other.p_type;
    }
    return *this;
}
dimn_t ScalarStream::col_count(dimn_t i) const noexcept
{
    RPY_CHECK(i<m_stream.size());
    return m_stream[i].size();
}
dimn_t ScalarStream::max_row_size() const noexcept
{
    if (m_stream.empty()) { return 0; }

    std::vector<dimn_t> tmp;
    tmp.reserve(m_stream.size());

    for (auto&& arr : m_stream ) {
        tmp.push_back(arr.size());
    }

    return *std::max_element(tmp.begin(), tmp.end());
}
ScalarArray ScalarStream::operator[](dimn_t row) const noexcept {
    RPY_CHECK(row < m_stream.size());
    return m_stream[row];
}
Scalar ScalarStream::operator[](std::pair<dimn_t, dimn_t> index) const noexcept {
    RPY_CHECK(index.first < m_stream.size());
    return m_stream[index.first][index.second];
}
void ScalarStream::set_ctype(const scalars::ScalarType* type) noexcept
{
    p_type = type;
}
void ScalarStream::reserve_size(dimn_t num_rows)
{
    m_stream.reserve(num_rows);
}
void ScalarStream::push_back(const ScalarArray& data)
{
    m_stream.push_back(data);
}

void ScalarStream::push_back(ScalarArray&& data)
{
    m_stream.push_back(std::move(data));
}
