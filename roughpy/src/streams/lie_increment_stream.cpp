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

#include "lie_increment_stream.h"

#include <roughpy/core/helpers.h>
#include <roughpy/scalars/key_scalar_array.h>
#include <roughpy/scalars/owned_scalar_array.h>
#include <roughpy/scalars/scalar_pointer.h>
#include <roughpy/scalars/scalar_type.h>
#include <roughpy/streams/lie_increment_stream.h>
#include <roughpy/streams/stream.h>

#include "args/kwargs_to_path_metadata.h"
#include "scalars/scalar_type.h"
#include "scalars/scalars.h"
#include "stream.h"

using namespace rpy;
using namespace rpy::python;
using namespace pybind11::literals;

static const char* LIE_INCR_STREAM_DOC
        = R"rpydoc(A basic stream type defined by a sequence of increments
of fixed size at specified time intervals.
)rpydoc";

void buffer_to_indices(
        std::vector<param_t>& indices, const py::buffer_info& info
)
{
    auto count = info.size;
    const auto* ptr = info.ptr;

    indices.resize(count);
    auto* dst = indices.data();
    if (info.format[0] == 'd') {
        memcpy(dst, ptr, count * sizeof(double));
    } else {
        auto conversion
                = scalars::get_conversion(py_buffer_to_type_id(info), "f64");
        conversion(
                scalars::ScalarPointer{nullptr, dst},
                scalars::ScalarPointer{nullptr, ptr}, count
        );
    }
}

static py::object lie_increment_stream_from_increments(
        const py::object& data, const py::kwargs& kwargs
)
{
    auto md = kwargs_to_metadata(kwargs);

    std::vector<param_t> indices;

    python::PyToBufferOptions options;
    options.type = md.scalar_type;
    options.max_nested = 2;
    options.allow_scalar = false;

    auto buffer = python::py_to_buffer(data, options);

    idimn_t increment_size = 0;
    idimn_t num_increments = 0;

    if (options.shape.empty()) {
        increment_size = static_cast<idimn_t>(buffer.size());
        num_increments = 1;
    } else if (options.shape.size() == 1) {
        increment_size = options.shape[0];
        num_increments = 1;
    } else {
        increment_size = options.shape[1];
        num_increments = options.shape[0];
    }

    if (md.scalar_type == nullptr) {
        if (options.type != nullptr) {
            md.scalar_type = options.type;
        } else {
            RPY_THROW(py::type_error, "unable to deduce suitable scalar type");
        }
    }

    RPY_CHECK(
            buffer.size()
            == static_cast<dimn_t>(increment_size * num_increments)
    );
    RPY_CHECK(md.scalar_type != nullptr);
    if (!md.ctx) {
        if (md.width == 0 || md.depth == 0) {
            RPY_THROW(py::value_error,
                    "either ctx or both width and depth must be specified"
            );
        }
        md.ctx = algebra::get_context(md.width, md.depth, md.scalar_type);
    }

    auto effective_support
            = intervals::RealInterval::right_unbounded(0.0, md.interval_type);

    if (kwargs.contains("indices")) {
        auto indices_arg = kwargs["indices"];

        if (py::isinstance<py::buffer>(indices_arg)) {
            auto info
                    = py::reinterpret_borrow<py::buffer>(indices_arg).request();
            buffer_to_indices(indices, info);
        } else if (py::isinstance<py::int_>(indices_arg)) {
            // Interpret this as a column in the data;
            auto icol = indices_arg.cast<idimn_t>();
            if (icol < 0) { icol += increment_size; }
            if (icol < 0 || icol >= increment_size) {
                RPY_THROW(py::value_error, "index out of bounds");
            }

            indices.reserve(num_increments);
            for (idimn_t i = 0; i < num_increments; ++i) {
                indices.push_back(static_cast<param_t>(
                        buffer[i * increment_size + icol].to_scalar_t()
                ));
            }
        } else if (py::isinstance<py::sequence>(indices_arg)) {
            indices = indices_arg.cast<std::vector<param_t>>();
        } else {
            RPY_THROW(py::type_error,"unexpected type provided to 'indices' "
                                 "argument");
        }

        if (!indices.empty()) {
            auto minmax = std::minmax_element(indices.begin(), indices.end());
            effective_support = intervals::RealInterval(
                    *minmax.first, *minmax.second + 1.0, md.interval_type
            );
        }
    }

    if (indices.empty()) {
        indices.reserve(num_increments);
        for (idimn_t i = 0; i < num_increments; ++i) {
            indices.emplace_back(i);
        }
    } else if (static_cast<idimn_t>(indices.size()) != num_increments) {
        RPY_THROW(py::value_error,"mismatch between number of rows in data and "
                              "number of indices");
    }

    auto result = streams::Stream(streams::LieIncrementStream(
            std::move(buffer).copy_or_move(), indices,
            {md.width, effective_support, md.ctx, md.scalar_type,
             md.vector_type ? *md.vector_type : algebra::VectorType::Dense,
             md.resolution}
    ));
    if (md.support) { result.restrict_to(*md.support); }

    if (options.cleanup) { options.cleanup(); }

    return py::reinterpret_steal<py::object>(
            python::RPyStream_FromStream(std::move(result))
    );
}

RPY_UNUSED static streams::Stream
lie_increment_path_from_values(const py::object& data, const py::kwargs& kwargs)
{
    RPY_THROW(std::runtime_error, "Not implemented");
}

void python::init_lie_increment_stream(py::module_& m)
{

    py::class_<streams::LieIncrementStream> klass(
            m, "LieIncrementStream", LIE_INCR_STREAM_DOC
    );

    klass.def_static(
            "from_increments", &lie_increment_stream_from_increments, "data"_a
    );
    //    klass.def_static("from_values", &lie_increment_path_from_values,
    //    "data"_a);
}
