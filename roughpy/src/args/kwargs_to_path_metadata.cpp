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

#include "kwargs_to_path_metadata.h"

#include <memory>

#include "algebra/context.h"
#include "scalars/scalar_type.h"

#include "numpy.h"
#include "parse_schema.h"

using namespace rpy;

python::PyStreamMetaData
python::kwargs_to_metadata(const pybind11::kwargs& kwargs)
{

    PyStreamMetaData md{
            0,                              // width
            0,                              // depth
            {},                             // support
            nullptr,                        // context
            nullptr,                        // scalar type
            {},                             // vector type
            0,                              // default resolution
            intervals::IntervalType::Clopen,// interval type
            nullptr                         // schema
    };

    streams::ChannelType ch_type;

    if (kwargs.contains("schema")) {
        auto schema = kwargs["schema"];
        if (py::isinstance<streams::StreamSchema>(schema)) {
            md.schema = schema.cast<std::shared_ptr<streams::StreamSchema>>();
        } else {
            md.schema = parse_schema(schema);
        }
    } else if (kwargs.contains("channel_types")) {
        auto channels = kwargs["channel_types"];

        // homogeneous channel types
        if (py::isinstance<streams::ChannelType>(channels)) {
            ch_type = channels.cast<streams::ChannelType>();
        } else if (py::isinstance<py::str>(channels)) {
            ch_type = string_to_channel_type(channels.cast<string>());

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
        ch_type = streams::ChannelType::Increment;
    }

    if (kwargs.contains("ctx")) {
        auto ctx = kwargs["ctx"];
        if (!py::isinstance(
                    ctx, reinterpret_cast<PyObject*>(&RPyContext_Type)
            )) {
            RPY_THROW(py::type_error, "expected a Context object");
        }
        md.ctx = python::ctx_cast(ctx.ptr());
        md.width = md.ctx->width();
        md.scalar_type = md.ctx->ctype();

        if (!md.schema) {
            md.schema = std::make_shared<streams::StreamSchema>();
            for (deg_t i = 0; i < md.width; ++i) {
                switch (ch_type) {
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

        auto schema_width = static_cast<deg_t>(md.schema->width());
        if (schema_width != md.width) {
            md.width = schema_width;
            md.ctx = md.ctx->get_alike(
                    md.width, md.ctx->depth(), md.scalar_type
            );
        }
    } else {
        if (md.schema) {
            md.width = static_cast<deg_t>(md.schema->width());
        } else if (kwargs.contains("width")) {
            md.width = kwargs["width"].cast<rpy::deg_t>();
        }

        if (kwargs.contains("depth")) {
            md.depth = kwargs["depth"].cast<rpy::deg_t>();
        }
        if (kwargs.contains("dtype")) {
            auto dtype = kwargs["dtype"];
#ifdef ROUGHPY_WITH_NUMPY
            if (py::isinstance<py::dtype>(dtype)) {
                md.scalar_type = npy_dtype_to_ctype(dtype);
            } else {
                md.scalar_type = py_arg_to_ctype(dtype);
            }
#else
            md.scalar_type = py_arg_to_ctype(dtype);
#endif
        }
    }

    if (kwargs.contains("vtype")) {
        md.vector_type = kwargs["vtype"].cast<algebra::VectorType>();
    }

    if (kwargs.contains("resolution")) {
        md.resolution = kwargs["resolution"].cast<resolution_t>();
    }

    if (kwargs.contains("support")) {
        auto support = kwargs["support"];
        if (!py::isinstance<intervals::Interval>(support)) {
            md.support = intervals::RealInterval(
                    support.cast<const intervals::Interval&>()
            );
        }
    }

    // TODO: Code for getting interval type

    return md;
}
