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
// Created by sam on 18/03/23.
//

#include "BaseStream.h"

using namespace rpy;
using namespace rpy::python;

static const char* STREAM_INTERFACE_DOC
        = R"rpydoc(The stream interface is the means by which one converts
an example of streaming data into a rough path.
)rpydoc";

void python::init_base_stream(py::module_& m)
{

    py::class_<streams::StreamInterface, PyBaseStream> klass(
            m, "StreamInterface", STREAM_INTERFACE_DOC
    );

    // TODO: Finish this off.
}

algebra::Lie PyBaseStream::log_signature_impl(
        const intervals::Interval& interval, const algebra::Context& ctx
) const
{
    PYBIND11_OVERRIDE_PURE(
            algebra::Lie, streams::StreamInterface, log_signature_impl,
            interval, ctx
    );
}
bool PyBaseStream::empty(const intervals::Interval& interval) const noexcept
{
    PYBIND11_OVERRIDE(bool, streams::StreamInterface, empty, interval);
}

algebra::Lie PyBaseStream::log_signature(
        const intervals::DyadicInterval& interval, resolution_t resolution,
        const algebra::Context& ctx
) const
{
    PYBIND11_OVERRIDE(
            algebra::Lie, streams::StreamInterface, log_signature, interval,
            resolution, ctx
    );
}
algebra::Lie PyBaseStream::log_signature(
        const intervals::Interval& interval, resolution_t resolution,
        const algebra::Context& ctx
) const
{
    PYBIND11_OVERRIDE(
            algebra::Lie, streams::StreamInterface, log_signature, interval,
            resolution, ctx
    );
}
algebra::FreeTensor PyBaseStream::signature(
        const intervals::Interval& interval, resolution_t resolution,
        const algebra::Context& ctx
) const
{
    PYBIND11_OVERRIDE(
            algebra::FreeTensor, streams::StreamInterface, signature, interval,
            resolution, ctx
    );
}
algebra::Lie PyBaseStream::log_signature(
        const intervals::Interval& interval, const algebra::Context& ctx
) const
{
    PYBIND11_OVERRIDE(
            algebra::Lie, streams::StreamInterface, log_signature, interval, ctx
    );
}
algebra::FreeTensor PyBaseStream::signature(
        const intervals::Interval& interval, const algebra::Context& ctx
) const
{
    PYBIND11_OVERRIDE(
            algebra::FreeTensor, streams::StreamInterface, signature, interval,
            ctx
    );
}
