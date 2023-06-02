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

#include "tick_stream.h"
#include "intervals/interval.h"
#include "roughpy/algebra/lie.h"

#include <roughpy/streams/tick_stream.h>
#include <roughpy/streams/stream.h>

#include "args/kwargs_to_path_metadata.h"
#include "scalars/scalar_type.h"
#include "scalars/scalars.h"
#include "streams/stream.h"

using namespace rpy;
using namespace pybind11::literals;

static const char* TICK_STREAM_DOC = R"rpydoc(Tick stream
)rpydoc";


static void populate_schema_dict(const std::shared_ptr<streams::StreamSchema>& schema, const py::dict& data) {
    for (auto&& [key, datum] : data) {
        streams::ChannelType type = rpy::streams::ChannelType::Increment;
        string label;
        if (hasattr(datum, "label")) {
            label = getattr(datum, "label").cast<string>();
        } else if (py::isinstance<py::dict>(datum) && datum.contains("label")) {
            label = datum["label"].cast<string>();
        }






    }
}

static std::shared_ptr<streams::StreamSchema> parse_schema(const py::object& object) {
    if (!py::isinstance<py::sequence>(object)) {
        throw py::value_error("tick data must be provided as a sequence of ticks");
    }

    std::shared_ptr<streams::StreamSchema> schema(new streams::StreamSchema);


    return schema;
}


static py::object construct(const py::object& data, const py::kwargs& kwargs) {

    auto pmd = python::kwargs_to_metadata(kwargs);

    std::vector<param_t> indices;

    python::PyToBufferOptions options;
    options.type = pmd.scalar_type;
    options.max_nested = 2;
    options.allow_scalar = false;

    auto buffer = python::py_to_buffer(data, options);

    scalars::ScalarStream stream(options.type);
    stream.reserve_size(options.shape.size());


    auto result = streams::Stream(
        streams::TickStream(
            std::move(stream),
            std::vector<const key_type*>(),
            std::move(indices),
            pmd.resolution,
            {
                pmd.width,
                pmd.support ? *pmd.support : intervals::RealInterval(0, 1),
                pmd.ctx,
                pmd.scalar_type,
                pmd.vector_type ? *pmd.vector_type : algebra::VectorType::Dense,
                pmd.resolution
            }
            )
        );

    return py::reinterpret_steal<py::object>(python::RPyStream_FromStream(std::move(result)));
}


void python::init_tick_stream(py::module_ &m) {

    py::class_<streams::TickStream> klass(m, "TickStream", TICK_STREAM_DOC);

    klass.def_static("from", &construct, "data"_a);

}
