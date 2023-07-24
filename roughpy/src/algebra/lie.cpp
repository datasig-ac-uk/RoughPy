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

#include "lie.h"

#include <pybind11/operators.h>

#include <roughpy/algebra/context.h>
#include <roughpy/algebra/lie.h>
#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_type.h>

#include "args/kwargs_to_vector_construction.h"
#include "args/numpy.h"
#include "scalars/scalars.h"

#include "lie_key.h"
#include "setup_algebra_type.h"

using namespace rpy;
using namespace rpy::algebra;
using namespace pybind11::literals;

static const char* LIE_DOC = R"edoc(
Element of the free Lie algebra.
)edoc";

static Lie construct_lie(py::object data, py::kwargs kwargs)
{
    auto helper = python::kwargs_to_construction_data(kwargs);

    python::PyToBufferOptions options;
    options.type = helper.ctype;
    options.allow_scalar = false;

    auto buffer = py_to_buffer(data, options);

    if (helper.ctype == nullptr) {
        if (options.type == nullptr) {
            RPY_THROW(py::value_error,"could not deduce an appropriate scalar_type"
            );
        }
        helper.ctype = options.type;
    }

    if (helper.width == 0 && buffer.size() > 0) {
        helper.width = static_cast<deg_t>(buffer.size());
    }

    if (!helper.ctx) {
        if (helper.width == 0 || helper.depth == 0) {
            RPY_THROW(py::value_error,
                    "you must provide either context or both width and depth"
            );
        }
        helper.ctx = get_context(helper.width, helper.depth, helper.ctype, {});
    }

    auto result = helper.ctx->construct_lie({std::move(buffer), helper.vtype});

    if (options.cleanup) { options.cleanup(); }

    return result;
}

void python::init_lie(py::module_& m)
{

    py::options options;
    options.disable_function_signatures();

    pybind11::class_<Lie> klass(m, "Lie", LIE_DOC);
    klass.def(py::init(&construct_lie), "data"_a = py::none());

    setup_algebra_type(klass);

    klass.def("__getitem__", [](const Lie& self, key_type key) {
        return self[key];
    });

    klass.def("__repr__", [](const Lie& self) {
        std::stringstream ss;
        ss << "Lie(width=" << *self.width() << ", depth=" << *self.depth();
        ss << ", ctype=" << self.coeff_type()->info().name << ')';
//        self->print(ss);
        return ss.str();
    });
}
