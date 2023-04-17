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

#ifndef ROUGHPY_SCALARS_SCALAR_ARRAY_H_
#define ROUGHPY_SCALARS_SCALAR_ARRAY_H_

#include "scalars_fwd.h"
#include "roughpy_scalars_export.h"

#include <cassert>



#include "scalar_pointer.h"

namespace rpy { namespace scalars {

class ROUGHPY_SCALARS_EXPORT ScalarArray : public ScalarPointer {

protected:
    dimn_t m_size = 0;
    bool m_owns_values = false;


public:

    ScalarArray() = default;

    explicit ScalarArray(const ScalarType *type)
        : ScalarPointer(type)
    {}
    ScalarArray(const ScalarType *type, void* data, dimn_t size)
        : ScalarPointer(type, data), m_size(size)
    {}
    ScalarArray(const ScalarType *type, const void* data, dimn_t size)
        : ScalarPointer(type, data), m_size(size)
    {}
    ScalarArray(ScalarPointer begin, dimn_t size)
        : ScalarPointer(begin), m_size(size) {}

    ScalarArray(const ScalarArray &other) = default;
    ScalarArray(ScalarArray &&other) noexcept;
    ScalarArray &operator=(const ScalarArray &other) = default;
    ScalarArray &operator=(ScalarArray &&other) noexcept;


    constexpr dimn_t size() const noexcept { return m_size; }
};
}}

#endif // ROUGHPY_SCALARS_SCALAR_ARRAY_H_
