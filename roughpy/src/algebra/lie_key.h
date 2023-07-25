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

#ifndef RPY_PY_ALGEBRA_LIE_KEY_H_
#define RPY_PY_ALGEBRA_LIE_KEY_H_

#include "roughpy_module.h"

#include <boost/container/small_vector.hpp>

#include <roughpy/algebra/context_fwd.h>
#include <roughpy/algebra/lie_basis.h>

#include "lie_letter.h"

namespace rpy {
namespace python {

class PyLieKey
{
public:
    using container_type = boost::container::small_vector<PyLieLetter, 2>;
    using basis_type = algebra::LieBasis;

private:
    container_type m_data;
    basis_type m_basis;

public:
    explicit PyLieKey(basis_type basis);
    explicit PyLieKey(basis_type basis, let_t letter);
    explicit PyLieKey(
            basis_type basis,
            const boost::container::small_vector_base<PyLieLetter>& data
    );
    explicit PyLieKey(basis_type basis, let_t left, let_t right);
    explicit PyLieKey(basis_type basis, let_t left, const PyLieKey& right);
    explicit PyLieKey(basis_type basis, const PyLieKey& left, const PyLieKey& right);
    explicit PyLieKey(algebra::LieBasis basis, key_type key);
    PyLieKey(const algebra::Context* ctx, key_type key);

    deg_t width() const noexcept { return m_basis.width(); }

    bool is_letter() const noexcept;
    let_t as_letter() const;
    string to_string() const;
    PyLieKey lparent() const;
    PyLieKey rparent() const;

    deg_t degree() const;

    bool equals(const PyLieKey& other) const noexcept;
};

PyLieKey
parse_lie_key(const algebra::LieBasis& basis,
              const py::args& args,
              const py::kwargs& kwargs);

void init_py_lie_key(py::module_& m);

}// namespace python
}// namespace rpy

#endif// RPY_PY_ALGEBRA_LIE_KEY_H_
