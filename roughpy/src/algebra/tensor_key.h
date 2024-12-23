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

#ifndef RPY_PY_ALGEBRA_TENSOR_KEY_H_
#define RPY_PY_ALGEBRA_TENSOR_KEY_H_

#include "roughpy_module.h"

#include "roughpy/core/macros.h"              // for RPY_NO_DISCARD
#include <roughpy/core/hash.h>
#include <roughpy/core/slice.h>
#include "roughpy/core/types.h"               // for deg_t, key_type, let_t

#include <roughpy/algebra/tensor_basis.h>

namespace rpy {
namespace python {
namespace maths {
template <typename I, typename E>
constexpr I power(I arg, E exponent) noexcept
{
    if (exponent == 0) { return I(1); }
    if (exponent == 1) { return arg; }
    auto recurse = power(arg, exponent / 2);
    return recurse * recurse * ((exponent & 1) == 1 ? arg : I(1));
}

template <typename I, typename B>
constexpr I log(I arg, B base) noexcept
{
    return (arg < base) ? I(0) : I(1) + log(arg / static_cast<I>(base), base);
}

}// namespace maths

class PyTensorKey
{
    key_type m_key;
    algebra::TensorBasis m_basis;

public:
    explicit PyTensorKey(algebra::TensorBasis basis, key_type key);

    PyTensorKey(algebra::TensorBasis basis, Slice<let_t> letters);

    explicit operator key_type() const noexcept;

    RPY_NO_DISCARD string to_string() const;
    RPY_NO_DISCARD PyTensorKey lparent() const;
    RPY_NO_DISCARD PyTensorKey rparent() const;
    RPY_NO_DISCARD bool is_letter() const;

    RPY_NO_DISCARD deg_t width() const;
    RPY_NO_DISCARD deg_t depth() const;
    RPY_NO_DISCARD algebra::TensorBasis basis() const { return m_basis; }
    RPY_NO_DISCARD pair<PyTensorKey, PyTensorKey> split_n(deg_t n) const;

    RPY_NO_DISCARD deg_t degree() const;
    RPY_NO_DISCARD std::vector<let_t> to_letters() const;

    RPY_NO_DISCARD PyTensorKey reverse() const;

    RPY_NO_DISCARD
    bool equals(const PyTensorKey& other) const noexcept;
    RPY_NO_DISCARD
    bool less(const PyTensorKey& other) const noexcept;



    friend hash_t hash_value(const PyTensorKey& key) noexcept;

};

hash_t hash_value(const PyTensorKey& key) noexcept;


RPY_NO_DISCARD PyTensorKey operator*(const PyTensorKey& lhs, const PyTensorKey& rhs);

void init_py_tensor_key(py::module_& m);

}// namespace python
}// namespace rpy

#endif// RPY_PY_ALGEBRA_TENSOR_KEY_H_
