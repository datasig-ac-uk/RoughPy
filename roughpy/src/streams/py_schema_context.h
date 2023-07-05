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
// Created by user on 04/07/23.
//

#ifndef ROUGHPY_ROUGHPY_SRC_STREAMS_PY_SCHEMA_CONTEXT_H_
#define ROUGHPY_ROUGHPY_SRC_STREAMS_PY_SCHEMA_CONTEXT_H_

#include <roughpy/streams/schema.h>

#include "roughpy_module.h"

#include "args/convert_timestamp.h"

namespace rpy {
namespace python {

class PySchemaContext : public streams::SchemaContext
{
    py::object m_dt_reference;
    optional<PyDateTimeConversionOptions> m_dt_conversion{};

public:
    explicit PySchemaContext(PyDateTimeConversionOptions conversion_options)
        : m_dt_reference(py::none()),
          m_dt_conversion(std::move(conversion_options))
    {}

    RPY_NO_DISCARD
    intervals::RealInterval
    convert_parameter_interval(const intervals::Interval& arg) const override;

    void set_reference_dt(py::object dt_reference);
    void set_dt_timescale(PyDateTimeResolution timescale);
};

}// namespace python
}// namespace rpy

#endif// ROUGHPY_ROUGHPY_SRC_STREAMS_PY_SCHEMA_CONTEXT_H_
