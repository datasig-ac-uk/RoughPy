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

//
// Created by user on 14/03/23.
//

#include "algebra.h"

#include "algebra_iterator.h"
#include "basis.h"
#include "free_tensor.h"
#include "lie.h"
#include "lie_key.h"
#include "lie_key_iterator.h"
#include "shuffle_tensor.h"
#include "tensor_key.h"
#include "tensor_key_iterator.h"
#include "free_multiply_funcs.h"

void rpy::python::init_algebra(pybind11::module_& m)
{

    py::enum_<algebra::VectorType>(m, "VectorType")
            .value("DenseVector", algebra::VectorType::Dense)
            .value("SparseVector", algebra::VectorType::Sparse)
            .export_values();

    init_py_tensor_key(m);
    init_py_lie_key(m);
    init_tensor_key_iterator(m);
    init_lie_key_iterator(m);
    init_basis(m);
    init_context(m);
    init_algebra_iterator(m);

    init_free_tensor(m);
    init_shuffle_tensor(m);
    init_lie(m);

    init_free_multiply_funcs(m);

}
