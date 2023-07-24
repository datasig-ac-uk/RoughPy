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
// Created by user on 12/04/23.
//

#include "brownian_stream.h"

#include <roughpy/streams/brownian_stream.h>

#include "args/kwargs_to_path_metadata.h"
#include "stream.h"

using namespace rpy;
using namespace streams;
using namespace pybind11::literals;

static const char* BROWNIAN_PATH_DOC = R"rpydoc(A Brownian motion stream.
)rpydoc";

static py::object
Brownian_from_generator(const py::args& args, const py::kwargs& kwargs)
{

    auto pmd = python::kwargs_to_metadata(kwargs);

    if (pmd.scalar_type == nullptr) {
        pmd.scalar_type = scalars::ScalarType::of<double>();
    }

    if (pmd.depth == 0) { pmd.depth = 2; }

    if (pmd.ctx == nullptr) {
        if (pmd.width == 0) {
            RPY_THROW(std::invalid_argument, "width must be provided");
        }
        pmd.ctx = algebra::get_context(
                pmd.width, pmd.depth, pmd.scalar_type, {}
        );
    }

    // TODO: Fix this up properly.

    streams::StreamMetadata md{
            pmd.width,
            pmd.support ? *pmd.support
                        : intervals::RealInterval(
                                -std::numeric_limits<param_t>::infinity(),
                                std::numeric_limits<param_t>::infinity()
                        ),
            pmd.ctx,
            pmd.scalar_type,
            pmd.vector_type ? *pmd.vector_type : algebra::VectorType::Dense,
            pmd.resolution};

    // TODO: Finish this

    BrownianStream inner(pmd.scalar_type->get_rng(), std::move(md));
    Stream stream(std::move(inner));

    return py::reinterpret_steal<py::object>(
            python::RPyStream_FromStream(std::move(stream))
    );
}

void rpy::python::init_brownian_stream(py::module_& m)
{

    py::class_<BrownianStream> klass(m, "BrownianStream", BROWNIAN_PATH_DOC);

    klass.def_static("with_generator", &Brownian_from_generator);
}
