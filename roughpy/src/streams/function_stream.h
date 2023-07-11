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

#ifndef RPY_PY_STREAMS_FUNCTION_PATH_H_
#define RPY_PY_STREAMS_FUNCTION_PATH_H_

#include "roughpy_module.h"

#include <roughpy/streams/dynamically_constructed_stream.h>

namespace rpy {
namespace python {

class RPY_NO_EXPORT FunctionStream
    : public streams::DynamicallyConstructedStream
{
    py::object m_fn;

public:
    enum FunctionValueType
    {
        Value,
        Increment
    };

private:
    FunctionValueType m_val_type;

public:
    FunctionStream(
            py::object fn, FunctionValueType val_type,
            streams::StreamMetadata md
    );

protected:
    algebra::Lie log_signature_impl(
            const intervals::Interval& interval, const algebra::Context& ctx
    ) const override;

    pair<Lie, Lie> compute_child_lie_increments(
            DyadicInterval left_di, DyadicInterval right_di,
            const Lie& parent_value
    ) const override;
};

void init_function_stream(py::module_& m);

}// namespace python
}// namespace rpy

#endif// RPY_PY_STREAMS_FUNCTION_PATH_H_
