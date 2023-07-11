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

//
// Created by user on 06/07/23.
//

#ifndef ROUGHPY_ROUGHPY_SRC_SCALARS_R_PY_MONOMIAL_H_
#define ROUGHPY_ROUGHPY_SRC_SCALARS_R_PY_MONOMIAL_H_

#include "roughpy_module.h"
#include <roughpy/scalars/scalars_fwd.h>

struct RPyMonomial {
    PyObject_VAR_HEAD rpy::scalars::monomial m_data;
};
extern PyTypeObject RPyMonomial_Type;

PyObject* PyMonomial_FromIndeterminate(rpy::scalars::indeterminate_type indet);
PyObject* PyMonomial_FromMonomial(rpy::scalars::monomial arg);

rpy::scalars::monomial PyMonomial_AsMonomial(PyObject* py_monomial);

struct RPyPolynomial {
    PyObject_VAR_HEAD rpy::scalars::rational_poly_scalar m_data;
};

extern PyTypeObject RPyPolynomial_Type;

inline bool RPyMonomial_Check(PyObject* obj)
{
    return Py_TYPE(obj) == &RPyMonomial_Type;
}

inline bool RPyPolynomial_Check(PyObject* obj)
{
    return Py_TYPE(obj) == &RPyPolynomial_Type;
}

inline const rpy::scalars::rational_poly_scalar&
RPyPolynomial_cast(PyObject* obj) noexcept
{
    return reinterpret_cast<RPyPolynomial*>(obj)->m_data;
}

namespace rpy {
namespace python {

void init_monomial(py::module_& m);

}// namespace python
}// namespace rpy

#endif// ROUGHPY_ROUGHPY_SRC_SCALARS_R_PY_MONOMIAL_H_
