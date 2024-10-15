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

#include "kwargs_to_path_metadata.h"
#include "parse_algebra_configuration.h"

#include "algebra/context.h"
#include "scalars/scalar_type.h"

#include "numpy.h"
#include "parse_schema.h"

#include <memory>
#include <cmath>

using namespace rpy;

python::PyStreamMetaData python::kwargs_to_metadata(pybind11::kwargs& kwargs)
{

    PyStreamMetaData md{
            0,                                   // width
            0,                                   // depth
            {},                                  // support
            nullptr,                             // context
            nullptr,                             // scalar type
            {},                                  // vector type
            {},                                  // default resolution
            intervals::IntervalType::Clopen,     // interval type
            nullptr,                             // schema
            rpy::streams::ChannelType::Increment,// default channel type
            false                                // include_param_as_data
    };

    streams::ChannelType ch_type;

    if (kwargs.contains("schema")) {
        auto schema = kwargs_pop(kwargs, "schema");
        if (py::isinstance<streams::StreamSchema>(schema)) {
            md.schema = schema.cast<std::shared_ptr<streams::StreamSchema>>();
        } else {
            md.schema = parse_schema(schema);
        }
        if (!md.schema->is_final()) { md.schema->finalize(0); }
    }

    if (kwargs.contains("channel_types")) {
        if (md.schema && md.schema->is_final()) {
            PyErr_WarnEx(
                    PyExc_RuntimeWarning,
                    "specifying both a finalized schema and "
                    "\"channel_types\" ignores "
                    "the \"channel_types\" argument",
                    1
            );
        }
        auto channels = kwargs_pop(kwargs, "channel_types");

        // homogeneous channel types
        if (py::isinstance<streams::ChannelType>(channels)) {
            md.default_channel_type = channels.cast<streams::ChannelType>();
        } else if (py::isinstance<py::str>(channels)) {
            md.default_channel_type
                    = string_to_channel_type(channels.cast<string>());

            // labelled heterogeneous channel types
        } else if (py::isinstance<py::dict>(channels)) {
            auto channel_dict = py::reinterpret_borrow<py::dict>(channels);
            md.schema = std::make_shared<streams::StreamSchema>();

            for (auto&& [label, item] : channel_dict) {
                switch (python::py_to_channel_type(
                        py::reinterpret_borrow<py::object>(item)
                )) {
                    case streams::ChannelType::Increment:
                        md.schema->insert_increment(label.cast<string>());
                        break;
                    case streams::ChannelType::Value:
                        md.schema->insert_value(label.cast<string>());
                        break;
                    case streams::ChannelType::Categorical:
                        md.schema->insert_categorical(label.cast<string>());
                        break;
                    case streams::ChannelType::Lie:
                        md.schema->insert_lie(label.cast<string>());
                        break;
                }
            }

            // unlabelled heterogeneous channel types
        } else if (py::isinstance<py::sequence>(channels)) {
            auto channel_list = py::reinterpret_borrow<py::sequence>(channels);
            md.schema = std::make_shared<streams::StreamSchema>();

            for (auto&& item : channel_list) {
                switch (python::py_to_channel_type(
                        py::reinterpret_borrow<py::object>(item)
                )) {
                    case streams::ChannelType::Increment:
                        md.schema->insert_increment("");
                        break;
                    case streams::ChannelType::Value:
                        md.schema->insert_value("");
                        break;
                    case streams::ChannelType::Categorical:
                        md.schema->insert_categorical("");
                        break;
                    case streams::ChannelType::Lie:
                        md.schema->insert_lie("");
                        break;
                }
            }
        }

        // Unset channel to, assume homogeneous increments
    } else {
        md.default_channel_type = streams::ChannelType::Increment;
    }

    if (!md.schema) {
        // No schema was given, but a width was given so construct an empty
        // Schema.
        md.schema = std::make_shared<streams::StreamSchema>();
    }

    if (kwargs.contains("include_time")
        && kwargs_pop(kwargs, "include_time").cast<bool>()) {

        if (md.schema->is_final()) {
            RPY_THROW(
                    py::value_error,
                    "cannot modify the provided schema "
                    "since it is finalized"
            );
        }

        md.schema->parametrization()->add_as_channel();
        md.include_param_as_data = true;
    }

    // Now parse the algebra config
    auto algebra_config = parse_algebra_configuration(kwargs);

    if (algebra_config.ctx) {
        md.ctx = std::move(algebra_config.ctx);
        md.width = md.ctx->width();
        md.depth = md.ctx->depth();
        md.scalar_type = md.ctx->ctype();
    } else {
        if (algebra_config.width) { md.width = *algebra_config.width; }
        if (algebra_config.depth) { md.depth = *algebra_config.depth; }
        if (algebra_config.scalar_type != nullptr) {
            md.scalar_type = algebra_config.scalar_type;
        }
    }

    if (md.schema->is_final()) {
        if (!algebra_config.width) {
            algebra_config.width = static_cast<deg_t>(md.schema->width());
            RPY_DBG_ASSERT(md.width == 0);
            md.width = *algebra_config.width;
        } else if (md.schema->width() != *algebra_config.width) {
            RPY_THROW(
                    py::value_error,
                    "specified width does not match the schema width"
            );
        }
    }

    // Additional information that will not affect the algebra config.
    if (kwargs.contains("vtype")) {
        md.vector_type
                = kwargs_pop(kwargs, "vtype").cast<algebra::VectorType>();
    }

    if (kwargs.contains("resolution")) {
        md.resolution = kwargs_pop(kwargs, "resolution").cast<resolution_t>();
    }

    if (kwargs.contains("support")) {
        auto support = kwargs_pop(kwargs, "support");
        if (!py::isinstance<intervals::Interval>(support)) {
            md.support = intervals::RealInterval(
                    support.cast<const intervals::Interval&>()
            );
        }
    }

    // TODO: Code for getting interval type

    return md;
}

resolution_t python::param_to_resolution(param_t accuracy) noexcept
{
    int exponent;
    frexp(accuracy, &exponent);
    return -std::min(5, exponent - 1);
}
