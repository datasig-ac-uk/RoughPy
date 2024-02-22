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

#include "parse_algebra_configuration.h"

using namespace rpy;

python::PyVectorConstructionHelper
python::kwargs_to_construction_data(pybind11::kwargs& kwargs)
{

    PyVectorConstructionHelper helper;

    auto algebra_configuration = python::parse_algebra_configuration(kwargs);

    if (algebra_configuration.ctx != nullptr) {
        helper.ctx = algebra_configuration.ctx;
        helper.width = helper.ctx->width();
        helper.depth = helper.ctx->depth();
        helper.ctype = helper.ctx->ctype();
        helper.ctype_requested = true;
    } else {
        if (algebra_configuration.width) {
            helper.width = *algebra_configuration.width;
        }
        if (algebra_configuration.depth) {
            helper.depth = *algebra_configuration.depth;
        } else {
            helper.depth = 2;
        }
        if (algebra_configuration.scalar_type != nullptr) {
            helper.ctype = algebra_configuration.scalar_type;
            helper.ctype_requested = true;
        }
    }

    if (kwargs.contains("vector_type")) {
        helper.vtype
                = kwargs_pop(kwargs, "vector_type").cast<algebra::VectorType>();
        helper.vtype_requested = true;
    }

    if (kwargs.contains("keys")) {
        const auto arg = kwargs_pop(kwargs, "keys");
        if (py::isinstance<key_type>(arg)) {
        } else if (py::isinstance<py::buffer>(arg)) {
            auto key_info = arg.cast<py::buffer>().request();
        }
    }

    if (helper.width != 0 && helper.depth != 0 && helper.ctype != nullptr) {
        helper.ctx = algebra::get_context(
                helper.width,
                helper.depth,
                helper.ctype
        );
    }

    // we should now have consumed all the the kwargs. Run the check
    python::check_for_excess_arguments(kwargs);

    return helper;
}
