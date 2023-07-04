// Copyright (c) 2023 RoughPy Developers. All rights reserved.
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

#ifndef RPY_PY_ALGEBRA_LIE_LETTER_H_
#define RPY_PY_ALGEBRA_LIE_LETTER_H_

#include "roughpy_module.h"

#include <iosfwd>

namespace rpy {
namespace python {

class PyLieLetter
{
    dimn_t m_data = 0;

    constexpr explicit PyLieLetter(dimn_t raw) : m_data(raw) {}

public:
    PyLieLetter() = default;

    static constexpr PyLieLetter from_letter(let_t letter)
    {
        return PyLieLetter(1 + (dimn_t(letter) << 1));
    }

    static constexpr PyLieLetter from_offset(dimn_t offset)
    {
        return PyLieLetter(offset << 1);
    }

    constexpr bool is_offset() const noexcept { return (m_data & 1) == 0; }

    explicit operator let_t() const noexcept { return let_t(m_data >> 1); }

    explicit constexpr operator dimn_t() const noexcept { return m_data >> 1; }

    friend std::ostream& operator<<(std::ostream& os, const PyLieLetter& let);
};

std::ostream& operator<<(std::ostream& os, const PyLieLetter& let);

}// namespace python
}// namespace rpy

#endif// RPY_PY_ALGEBRA_LIE_LETTER_H_
