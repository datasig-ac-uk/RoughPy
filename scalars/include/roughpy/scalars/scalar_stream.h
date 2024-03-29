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

#ifndef ROUGHPY_SCALARS_SCALAR_STREAM_H_
#define ROUGHPY_SCALARS_SCALAR_STREAM_H_


#include "scalars_fwd.h"

#include <boost/container/small_vector.hpp>
#include <vector>

namespace rpy { namespace scalars {

class ROUGHPY_SCALARS_EXPORT ScalarStream {
protected:
    std::vector<ScalarArray> m_stream;
    const ScalarType* p_type;

public:
    RPY_NO_DISCARD const ScalarType* type() const noexcept { return p_type; }

    ScalarStream();
    ScalarStream(const ScalarStream& other);
    ScalarStream(ScalarStream&& other) noexcept;

    explicit ScalarStream(const ScalarType* type);
    ScalarStream(ScalarArray base, std::vector<dimn_t> shape);

    ScalarStream(
            std::vector<ScalarArray>&& stream,
            dimn_t row_elts,
            const ScalarType* type
    )
        : m_stream(std::move(stream)),
          p_type(type)
    {}

    ScalarStream& operator=(const ScalarStream& other);
    ScalarStream& operator=(ScalarStream&& other) noexcept;

    RPY_NO_DISCARD dimn_t col_count(dimn_t i = 0) const noexcept;
    RPY_NO_DISCARD dimn_t row_count() const noexcept { return m_stream.size(); }

    RPY_NO_DISCARD dimn_t max_row_size() const noexcept;

    RPY_NO_DISCARD ScalarArray operator[](dimn_t row) const noexcept;
    RPY_NO_DISCARD Scalar operator[](std::pair<dimn_t, dimn_t> index
    ) const noexcept;

    void set_ctype(const scalars::ScalarType* type) noexcept;

    void reserve_size(dimn_t num_rows);

    void push_back(const ScalarArray& data);
    void push_back(ScalarArray&& data);
};

}}

#endif // ROUGHPY_SCALARS_SCALAR_STREAM_H_
