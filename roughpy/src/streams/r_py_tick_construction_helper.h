// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
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
// Created by user on 22/06/23.
//

#ifndef ROUGHPY_ROUGHPY_SRC_STREAMS_R_PY_TICK_CONSTRUCTION_HELPER_H
#define ROUGHPY_ROUGHPY_SRC_STREAMS_R_PY_TICK_CONSTRUCTION_HELPER_H

#include "roughpy_module.h"

#include "args/convert_timestamp.h"
#include <roughpy/core/macros.h>
#include <roughpy/streams/schema.h>
#include <roughpy/streams/stream.h>

namespace rpy {
namespace python {

struct RPy_Tick {
    string label;
    param_t timestamp;
    py::object data;
    streams::ChannelType type;
};

class RPyTickConstructionHelper
{
    std::vector<RPy_Tick> m_ticks;
    std::shared_ptr<streams::StreamSchema> p_schema;
    bool b_schema_only;
    py::object m_reference_time;

    PyDateTimeConversionOptions m_time_conversion_options;

private:
    void add_increment_to_schema(string label, const py::kwargs& kwargs);
    void add_value_to_schema(string label, const py::kwargs& kwargs);
    void add_categorical_to_schema(
            string label, py::object variant, const py::kwargs& kwargs
    );

    RPY_NO_RETURN
    void fail_timestamp_none();

    RPY_NO_RETURN
    void fail_data_none();

public:
    RPyTickConstructionHelper();
    explicit RPyTickConstructionHelper(bool schema_only);
    explicit RPyTickConstructionHelper(
            std::shared_ptr<streams::StreamSchema> schema, bool schema_only
    );

private:
    void add_tick(
            string label, py::object timestamp, py::object data,
            streams::ChannelType type, const py::kwargs& kwargs
    );

public:
    void add_increment(
            const py::str& label, py::object timestamp, py::object data,
            const py::kwargs& kwargs
    );
    void add_value(
            const py::str& label, py::object timestamp, py::object data,
            const py::kwargs& kwargs
    );
    void add_categorical(
            const py::str& label, py::object timestamp, py::object variant,
            const py::kwargs& kwargs
    );
    //    void add_lie(const py::str& label, const py::kwargs& kwargs);

    const std::shared_ptr<streams::StreamSchema>& schema() const noexcept
    {
        return p_schema;
    }
    const std::vector<RPy_Tick>& ticks() const noexcept { return m_ticks; }
};

void init_tick_construction_helper(py::module_& m);

}// namespace python
}// namespace rpy

#endif// ROUGHPY_ROUGHPY_SRC_STREAMS_R_PY_TICK_CONSTRUCTION_HELPER_H
