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

#include "function_stream.h"

#include <roughpy/algebra/lie.h>

#include "algebra/context.h"
#include "args/kwargs_to_path_metadata.h"
#include "stream.h"

using namespace rpy;
using namespace pybind11::literals;

static const char* FUNC_STREAM_DOC
        = R"rpydoc(A stream generated dynamically by calling a function.
)rpydoc";

python::FunctionStream::FunctionStream(
        py::object fn, python::FunctionStream::FunctionValueType val_type,
        streams::StreamMetadata md
)
    : DynamicallyConstructedStream(std::move(md)), m_fn(std::move(fn)),
      m_val_type(val_type)
{}

algebra::Lie python::FunctionStream::log_signature_impl(
        const intervals::Interval& interval, const algebra::Context& ctx
) const
{
    py::gil_scoped_acquire gil;

    auto pyctx = py::reinterpret_steal<py::object>(
            python::RPyContext_FromContext(&ctx)
    );

    if (m_val_type == Increment) {
        return m_fn(interval, pyctx).cast<algebra::Lie>();
    }

    auto finf = m_fn(interval.inf(), pyctx).cast<algebra::Lie>();
    auto fsup = m_fn(interval.sup(), pyctx).cast<algebra::Lie>();

    return fsup.sub(finf);
}
pair<algebra::Lie, algebra::Lie>
python::FunctionStream::compute_child_lie_increments(
        streams::DynamicallyConstructedStream::DyadicInterval left_di,
        streams::DynamicallyConstructedStream::DyadicInterval right_di,
        const streams::DynamicallyConstructedStream::Lie& parent_value
) const
{
    const auto& md = metadata();

    return {log_signature_impl(left_di, *md.default_context),
            log_signature_impl(right_di, *md.default_context)};
}

static py::object from_function(py::object fn, py::kwargs kwargs)
{
    auto pmd = python::kwargs_to_metadata(kwargs);
    if (pmd.ctx == nullptr && pmd.width != 0 && pmd.depth != 0
        && pmd.scalar_type != nullptr) {
        pmd.ctx = algebra::get_context(
                pmd.width, pmd.depth, pmd.scalar_type, {}
        );
    }

    // TODO: Fix this up properly.

    intervals::RealInterval effective_support =
            intervals::RealInterval::unbounded();
    if (pmd.support) {
        effective_support = *pmd.support;
    }

    streams::StreamMetadata md{
            pmd.width,
            effective_support,
            pmd.ctx,
            pmd.scalar_type,
            pmd.vector_type ? *pmd.vector_type : algebra::VectorType::Dense,
            pmd.resolution};

    PyObject* stream = python::RPyStream_FromStream(
            streams::Stream(python::FunctionStream(
                    std::move(fn), python::FunctionStream::Value, std::move(md)
            ))
    );
    return py::reinterpret_steal<py::object>(stream);
}

void python::init_function_stream(py::module_& m)
{

    py::class_<FunctionStream> klass(m, "FunctionStream", FUNC_STREAM_DOC);

    klass.def_static("from_function", from_function, "function"_a);
}
