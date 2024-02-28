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

#ifndef ROUGHPY_SCALARS_KEY_SCALAR_STREAM_H_
#define ROUGHPY_SCALARS_KEY_SCALAR_STREAM_H_

#include "scalar_stream.h"
#include "scalars_fwd.h"

namespace rpy {
namespace scalars {

class ROUGHPY_SCALARS_EXPORT KeyScalarStream : public ScalarStream
{
    std::vector<const key_type*> m_key_stream;

public:
    KeyScalarStream();
    KeyScalarStream(const ScalarType* type);
    KeyScalarStream(const KeyScalarStream&);
    KeyScalarStream(KeyScalarStream&&) noexcept;

    KeyScalarStream& operator=(const KeyScalarStream&);
    KeyScalarStream& operator=(KeyScalarStream&&) noexcept;

    RPY_NO_DISCARD KeyScalarArray operator[](dimn_t row) const noexcept;

    void reserve_size(dimn_t num_rows);

    void push_back(
            const ScalarArray& scalar_data,
            const key_type* key_ptr = nullptr
    );
    void push_back(ScalarArray&& scalar_data, const key_type* key_ptr=nullptr);
};

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_KEY_SCALAR_STREAM_H_
