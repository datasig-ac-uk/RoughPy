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
// Created by user on 02/06/23.
//

#include "schema.h"
#include "args/parse_schema.h"
#include <roughpy/streams/schema.h>

#include "r_py_tick_construction_helper.h"

using namespace rpy;
using namespace rpy::python;
using namespace rpy::streams;
using namespace pybind11::literals;

namespace {

// class PyChannelItem {
//     StreamChannel *ptr;
//
// public:
//     PyChannelItem(StreamChannel &item) : ptr(&item) {}
// };

inline void init_channel_item(py::module_& m)
{

    py::enum_<ChannelType>(m, "ChannelType")
            .value("IncrementChannel", ChannelType::Increment)
            .value("ValueChannel", ChannelType::Value)
            .value("CategoricalChannel", ChannelType::Categorical)
            .value("LieChannel", ChannelType::Lie)
            .export_values();

    py::class_<StreamChannel> cls(m, "StreamChannel");
}

std::shared_ptr<StreamSchema>
schema_from_data(const py::object& data, const py::kwargs& kwargs)
{

    py::object parser;

    auto schema = std::make_shared<StreamSchema>();

    if (kwargs.contains("parser")) {
        parser = kwargs["parser"](schema, true);
    } else {
        auto tick_helpers_mod
                = py::module_::import("roughpy.streams.tick_stream");
        parser = tick_helpers_mod.attr("StandardTickDataParser")(schema, true);
    }

    parser.attr("parse_data")(data);
    auto& helper
            = parser.attr("helper").cast<python::RPyTickConstructionHelper&>();
    return helper.schema();
}

}// namespace

void rpy::python::init_schema(py::module_& m)
{

    init_channel_item(m);

    py::class_<StreamSchema, std::shared_ptr<StreamSchema>> cls(
            m, "StreamSchema"
    );

    cls.def_static("from_data", &schema_from_data, "data"_a);
    cls.def_static("parse", &parse_schema, "schema"_a);

    cls.def("width", &StreamSchema::width);

    cls.def(
            "insert_increment",
            [](StreamSchema* schema, string label) {
                return &schema->insert_increment(std::move(label));
            },
            "label"_a, py::return_value_policy::reference_internal
    );

    cls.def(
            "insert_value",
            [](StreamSchema* schema, string label) {
                return &schema->insert_value(std::move(label));
            },
            "label"_a, py::return_value_policy::reference_internal
    );

    cls.def(
            "insert_categorical",
            [](StreamSchema* schema, string label) {
                return &schema->insert_categorical(std::move(label));
            },
            "label"_a, py::return_value_policy::reference_internal
    );

    cls.def("get_labels", [](const StreamSchema* schema) {
        py::list labels(schema->width());
        auto* plist = labels.ptr();
        py::ssize_t i = 0;
        for (auto&& item : *schema) {
            switch (item.second.type()) {
                case streams::ChannelType::Increment:
                    PyList_SET_ITEM(
                            plist, i++, PyUnicode_FromString(item.first.c_str())
                    );
                    break;
                case streams::ChannelType::Value:
                    if (item.second.is_lead_lag()) {
                        PyList_SET_ITEM(
                                plist, i++,
                                PyUnicode_FromString(
                                        (item.first
                                         + item.second.label_suffix(0))
                                                .c_str()
                                )
                        );
                        PyList_SET_ITEM(
                                plist, i++,
                                PyUnicode_FromString(
                                        (item.first
                                         + item.second.label_suffix(1))
                                                .c_str()
                                )
                        );
                    } else {
                        PyList_SET_ITEM(
                                plist, i++,
                                PyUnicode_FromString(item.first.c_str())
                        );
                    }
                    break;
                case streams::ChannelType::Categorical: {
                    auto nvariants = item.second.num_variants();
                    for (dimn_t idx = 0; idx < nvariants; ++idx) {
                        PyList_SET_ITEM(
                                plist, i++,
                                PyUnicode_FromString(
                                        (item.first
                                         + item.second.label_suffix(idx))
                                                .c_str()
                                )
                        );
                    }
                    break;
                }
                case streams::ChannelType::Lie: break;
            }
        }
        return labels;
    });
}
