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

//
// Created by user on 22/06/23.
//

#include "r_py_tick_construction_helper.h"

#include <roughpy/streams/tick_stream.h>

#include "args/convert_timestamp.h"
#include "args/parse_schema.h"

#include "py_parametrization.h"
#include <algorithm>

using namespace rpy;
using namespace rpy::streams;
using namespace pybind11::literals;

python::RPyTickConstructionHelper::RPyTickConstructionHelper()
    : m_ticks(), p_schema(new StreamSchema), b_schema_only(false),
      m_reference_time(py::none()),
      m_time_conversion_options{PyDateTimeResolution::Seconds}
{}
python::RPyTickConstructionHelper::RPyTickConstructionHelper(bool schema_only)
    : m_ticks(), p_schema(new StreamSchema), b_schema_only(schema_only),
      m_reference_time(py::none()),
      m_time_conversion_options{PyDateTimeResolution::Seconds}
{}

python::RPyTickConstructionHelper::RPyTickConstructionHelper(
        std::shared_ptr<streams::StreamSchema> schema, bool schema_only
)
    : m_ticks(), p_schema(std::move(schema)), b_schema_only(schema_only),
      m_reference_time(py::none()),
      m_time_conversion_options{PyDateTimeResolution::Seconds}
{
    if (!p_schema->is_final() && p_schema->parametrization() == nullptr) {
        p_schema->init_context<PySchemaContext>(m_time_conversion_options);
    }
    RPY_CHECK(!schema_only || !p_schema->is_final());
}

void python::RPyTickConstructionHelper::add_increment_to_schema(
        string label, const py::kwargs& RPY_UNUSED_VAR kwargs
)
{
    p_schema->insert_increment(std::move(label));
}
void python::RPyTickConstructionHelper::add_value_to_schema(
        string label, const py::kwargs& RPY_UNUSED_VAR kwargs
)
{
    p_schema->insert_value(std::move(label));
}
void python::RPyTickConstructionHelper::add_categorical_to_schema(
        string label, py::object variant,
        const py::kwargs& RPY_UNUSED_VAR kwargs
)
{
    auto& channel = p_schema->insert_categorical(std::move(label));
    if (!variant.is_none()) { channel.insert_variant(variant.cast<string>()); }
}

void python::RPyTickConstructionHelper::fail_timestamp_none()
{
    RPY_THROW(
            py::value_error, "timestamp cannot be None when constructing stream"
    );
}
void python::RPyTickConstructionHelper::fail_data_none()
{
    RPY_THROW(py::value_error, "data cannot be None when constructing stream");
}

const std::vector<python::RPy_Tick>&
python::RPyTickConstructionHelper::ticks() noexcept
{
    std::sort(
            m_ticks.begin(), m_ticks.end(),
            [](const RPy_Tick& ltick, const RPy_Tick& rtick) {
                return ltick.timestamp < rtick.timestamp
                        || (ltick.timestamp == rtick.timestamp
                            && ltick.precedence < rtick.precedence);
            }
    );
    return m_ticks;
}

void python::RPyTickConstructionHelper::add_tick(
        string label, py::object timestamp, py::object data,
        streams::ChannelType type, const py::kwargs& RPY_UNUSED_VAR kwargs
)
{

    if (b_schema_only) {
        // Do Nothing
    } else if (timestamp.is_none()) {
        fail_timestamp_none();
    } else if (data.is_none()) {
        fail_data_none();
    } else {
        if (m_reference_time.is_none()) { m_reference_time = timestamp; }
        const auto param = python::convert_delta_from_datetimes(
                timestamp, m_reference_time, m_time_conversion_options
        );

        m_ticks.push_back(
                {label, param, std::move(data), type, m_precedence[param]++}
        );
    }
}

void python::RPyTickConstructionHelper::add_increment(
        const py::str& label, py::object timestamp, py::object data,
        const py::kwargs& kwargs
)
{
    auto lbl = label.cast<string>();
    if (!p_schema->is_final()) { add_increment_to_schema(lbl, kwargs); }

    add_tick(
            std::move(lbl), std::move(timestamp), std::move(data),
            streams::ChannelType::Increment, kwargs
    );
}
void python::RPyTickConstructionHelper::add_value(
        const py::str& label, py::object timestamp, py::object data,
        const py::kwargs& kwargs
)
{
    auto lbl = label.cast<string>();
    if (!p_schema->is_final()) { add_value_to_schema(lbl, kwargs); }

    add_tick(
            std::move(lbl), std::move(timestamp), std::move(data),
            streams::ChannelType::Value, kwargs
    );
}
void python::RPyTickConstructionHelper::add_categorical(
        const py::str& label, py::object timestamp, py::object variant,
        const py::kwargs& kwargs
)
{
    auto lbl = label.cast<string>();
    if (!p_schema->is_final()) {
        add_categorical_to_schema(lbl, variant, kwargs);
    }

    add_tick(
            label.cast<string>(), std::move(timestamp), std::move(variant),
            streams::ChannelType::Categorical, kwargs
    );
}

void python::RPyTickConstructionHelper::add_time_channel() noexcept {
    RPY_DBG_ASSERT(p_schema->parametrization() != nullptr);
    p_schema->parametrization()->add_as_channel();
}

void python::init_tick_construction_helper(py::module_& m)
{
    using helper_t = python::RPyTickConstructionHelper;

    static const char* TICK_STREAM_CONSTRUCTOR_DOC
            = R"eadoc(
            Helps the stream figure out what its schema should be.
            A means of constructing the schema for a stream.
            )eadoc";

    py::class_<helper_t> klass(m, "TickStreamConstructionHelper", TICK_STREAM_CONSTRUCTOR_DOC);

    klass.def(py::init<>());
    klass.def(py::init<bool>(), "schema_only"_a);
    klass.def(
            py::init<std::shared_ptr<StreamSchema>, bool>(), "schema"_a,
            "schema_only"_a = false
    );
    klass.def("add_time_channel", &helper_t::add_time_channel);

    klass.def(
            "add_increment", &helper_t::add_increment, "label"_a,
            "timestamp"_a = py::none(), "data"_a = py::none()
    );
    klass.def(
            "add_value", &helper_t::add_value, "label"_a,
            "timestamp"_a = py::none(), "data"_a = py::none()
    );
    klass.def(
            "add_categorical", &helper_t::add_categorical, "label"_a,
            "timestamp"_a = py::none(), "variant"_a = py::none()
    );
}
