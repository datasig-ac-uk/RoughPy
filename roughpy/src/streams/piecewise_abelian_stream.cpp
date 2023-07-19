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
// Created by user on 18/03/23.
//

#include "piecewise_abelian_stream.h"

#include <roughpy/streams/piecewise_abelian_stream.h>
#include <roughpy/streams/stream.h>

#include "args/kwargs_to_path_metadata.h"
#include "stream.h"

using namespace rpy;
using namespace pybind11::literals;

static const char* PW_LIE_STREAM_DOC
        = R"rpydoc(A stream formed of a sequence of interval-Lie pairs.
)rpydoc";

static py::object construct_piecewise_lie_stream(
        std::vector<std::pair<intervals::RealInterval, algebra::Lie>> lies,
        const py::kwargs& kwargs
)
{

    auto pmd = python::kwargs_to_metadata(kwargs);

    if (!pmd.ctx) {
        if (pmd.width == 0 || pmd.depth == 0 || !pmd.scalar_type) {
            RPY_THROW(py::value_error, "cannot deduce width/depth/ctype");
        }
        pmd.ctx = algebra::get_context(pmd.width, pmd.depth, pmd.scalar_type);
    }
    // TODO: make this more robust

    using nl = std::numeric_limits<param_t>;
    param_t a = nl::infinity();
    param_t b = -nl::infinity();
    for (auto& piece : lies) {
        if (piece.first.inf() < a) { a = piece.first.inf(); }
        if (piece.first.sup() > b) { b = piece.first.sup(); }
    }

    pmd.support = intervals::RealInterval(a, b);

    streams::Stream result(streams::PiecewiseAbelianStream(
            std::move(lies),
            {pmd.width,
             pmd.support ? *pmd.support : intervals::RealInterval(0, 1),
             pmd.ctx, pmd.scalar_type,
             pmd.vector_type ? *pmd.vector_type : algebra::VectorType::Dense,
             pmd.resolution}
    ));

    return py::reinterpret_steal<py::object>(
            python::RPyStream_FromStream(std::move(result))
    );
}

void python::init_piecewise_lie_stream(py::module_& m)
{

    py::class_<streams::PiecewiseAbelianStream> klass(
            m, "PiecewiseAbelianStream", PW_LIE_STREAM_DOC
    );

    klass.def_static("construct", &construct_piecewise_lie_stream);
}
