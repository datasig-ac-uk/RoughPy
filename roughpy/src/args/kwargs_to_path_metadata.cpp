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

#include "kwargs_to_path_metadata.h"

#include "algebra/context.h"
#include "scalars/scalar_type.h"

#include "numpy.h"

using namespace rpy;

python::PyStreamMetaData python::kwargs_to_metadata(const pybind11::kwargs &kwargs) {

    PyStreamMetaData md{
        0,      // width
        0,      // depth
        {},     // support
        nullptr,// context
        nullptr,// scalar type
        {},     // vector type
        0       // default resolution
    };

    if (kwargs.contains("ctx")) {
        auto ctx = kwargs["ctx"];
        if (!py::isinstance(ctx, reinterpret_cast<PyObject *>(&RPyContext_Type))) {
            throw py::type_error("expected a Context object");
        }
        md.ctx = python::ctx_cast(ctx.ptr());
        md.width = md.ctx->width();
        md.scalar_type = md.ctx->ctype();
    } else {

        if (kwargs.contains("width")) {
            md.width = kwargs["width"].cast<rpy::deg_t>();
        }
        if (kwargs.contains("depth")) {
            md.depth = kwargs["depth"].cast<rpy::deg_t>();
        }
        if (kwargs.contains("ctype")) {
            md.scalar_type = python::py_arg_to_ctype(kwargs["ctype"]);
        }
#ifdef ROUGHPY_WITH_NUMPY
        else if (kwargs.contains("dtype")) {
            auto dtype = kwargs["dtype"];
            if (py::isinstance<py::dtype>(dtype)) {
                md.scalar_type = npy_dtype_to_ctype(dtype);
            } else {
                md.scalar_type = py_arg_to_ctype(dtype);
            }
        }
#endif
    }

    if (kwargs.contains("vtype")) {
        md.vector_type = kwargs["vtype"].cast<algebra::VectorType>();
    }

    return md;
}
