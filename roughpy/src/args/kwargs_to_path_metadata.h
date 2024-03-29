// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
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

#ifndef RPY_PY_ARGS_KWARGS_TO_PATH_METADATA_H_
#define RPY_PY_ARGS_KWARGS_TO_PATH_METADATA_H_

#include "roughpy_module.h"

#include <roughpy/streams/schema.h>
#include <roughpy/streams/stream_base.h>

namespace rpy {
namespace python {

struct PyStreamMetaData {
    deg_t width;
    deg_t depth;
    optional<intervals::RealInterval> support;
    algebra::context_pointer ctx;
    const scalars::ScalarType* scalar_type;
    optional<algebra::VectorType> vector_type;
    optional<resolution_t> resolution;
    intervals::IntervalType interval_type;
    std::shared_ptr<streams::StreamSchema> schema;
    streams::ChannelType default_channel_type;
    bool include_param_as_data;
};

PyStreamMetaData kwargs_to_metadata(py::kwargs& kwargs);

resolution_t param_to_resolution(param_t accuracy) noexcept;

}// namespace python
}// namespace rpy

#endif// RPY_PY_ARGS_KWARGS_TO_PATH_METADATA_H_
