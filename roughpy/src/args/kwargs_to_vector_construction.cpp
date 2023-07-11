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

#include "kwargs_to_vector_construction.h"

#include "algebra/context.h"
#include "scalars/scalar_type.h"

using namespace rpy;

python::PyVectorConstructionHelper
python::kwargs_to_construction_data(const pybind11::kwargs& kwargs)
{

    PyVectorConstructionHelper helper;

    if (kwargs.contains("ctx")) {
        helper.ctx = python::ctx_cast(kwargs["ctx"].ptr());
        helper.width = helper.ctx->width();
        helper.depth = helper.ctx->depth();
        helper.ctype = helper.ctx->ctype();
        helper.ctype_requested = true;
    } else {
        if (kwargs.contains("dtype")) {
            helper.ctype = py_arg_to_ctype(kwargs["dtype"]);
            helper.ctype_requested = true;
        }

        if (kwargs.contains("depth")) {
            helper.depth = kwargs["depth"].cast<deg_t>();
        } else {
            helper.depth = 2;
        }

        if (kwargs.contains("width")) {
            helper.width = kwargs["width"].cast<deg_t>();
        }
    }

    if (kwargs.contains("vector_type")) {
        helper.vtype = kwargs["vector_type"].cast<algebra::VectorType>();
        helper.vtype_requested = true;
    }

    if (kwargs.contains("keys")) {
        const auto& arg = kwargs["keys"];
        if (py::isinstance<key_type>(arg)) {
        } else if (py::isinstance<py::buffer>(arg)) {
            auto key_info = arg.cast<py::buffer>().request();
        }
    }

    if (helper.width != 0 && helper.depth != 0 && helper.ctype != nullptr) {
        helper.ctx = algebra::get_context(
                helper.width, helper.depth, helper.ctype
        );
    }

    return helper;
}
