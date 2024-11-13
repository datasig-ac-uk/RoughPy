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
// Created by user on 02/06/23.
//

#include "schema.h"

#include "roughpy/core/check.h"                              // for throw_ex...
#include "roughpy/core/macros.h"                             // for RPY_LOCAL
#include "roughpy/core/types.h"                              // for string

#include "args/parse_schema.h"
#include <roughpy/streams/schema.h>

#include "r_py_tick_construction_helper.h"

using namespace rpy;
using namespace rpy::python;
using namespace rpy::streams;
using namespace pybind11::literals;

namespace rpy {
namespace python {
class RPY_LOCAL RPyStreamChannel : public StreamChannel
{
public:
    using StreamChannel::StreamChannel;

    dimn_t num_variants() const override
    {
        PYBIND11_OVERRIDE(dimn_t, StreamChannel, num_variants);
    }
    string label_suffix(dimn_t variant_no) const override
    {
        PYBIND11_OVERRIDE(string, StreamChannel, label_suffix, variant_no);
    }
    dimn_t variant_id_of_label(string_view label) const override
    {
        PYBIND11_OVERRIDE(dimn_t, StreamChannel, variant_id_of_label, label);
    }
    void
    set_lie_info(deg_t width, deg_t depth, algebra::VectorType vtype) override
    {
        if (type() == ChannelType::Lie) {
            PYBIND11_OVERRIDE(
                    void, StreamChannel, set_lie_info, width, depth, vtype
            );
        } else {
            RPY_THROW(
                    std::runtime_error,
                    "set_lie_info should only be used for Lie-type channels"
            );
        }
    }

    StreamChannel& add_variant(string variant_label) override
    {
        if (type() == ChannelType::Categorical) {
            PYBIND11_OVERRIDE(
                    StreamChannel&, StreamChannel, add_variant, variant_label
            );
        } else {
            RPY_THROW(
                    std::runtime_error,
                    "only categorical channels can have variants"
            );
        }
    }
    StreamChannel& insert_variant(string variant_label) override
    {
        if (type() == ChannelType::Categorical) {
            PYBIND11_OVERRIDE(
                    StreamChannel&, StreamChannel, insert_variant, variant_label
            );
        } else {
            RPY_THROW(
                    std::runtime_error,
                    "only categorical channels can have variants"
            );
        }
    }
    const std::vector<string>& get_variants() const override
    {
        if (type() == ChannelType::Categorical) {
            PYBIND11_OVERRIDE(const std::vector<string>&, StreamChannel, get_variants);
        } else {
            RPY_THROW(
                    std::runtime_error,
                    "only categorical channels can have variants"
            );
        }
    }
};


class RPY_LOCAL RPyLeadLaggableChannel : public LeadLaggableChannel
{
public:

    using LeadLaggableChannel::LeadLaggableChannel;

    dimn_t num_variants() const override
    {
        PYBIND11_OVERRIDE(dimn_t, LeadLaggableChannel, num_variants);
    }
    string label_suffix(dimn_t variant_no) const override
    {
        PYBIND11_OVERRIDE(string, LeadLaggableChannel, label_suffix, variant_no);
    }
    dimn_t variant_id_of_label(string_view label) const override
    {
        PYBIND11_OVERRIDE(dimn_t, LeadLaggableChannel, variant_id_of_label, label);
    }
    const std::vector<string>& get_variants() const override
    {
        PYBIND11_OVERRIDE(const std::vector<string>&, LeadLaggableChannel, get_variants);
    }
    void set_lead_lag(bool new_value) override
    {
        PYBIND11_OVERRIDE(void, LeadLaggableChannel, set_lead_lag, new_value);
    }
    bool is_lead_lag() const override
    {
        PYBIND11_OVERRIDE(bool, LeadLaggableChannel, is_lead_lag);
    }
    void
    set_lie_info(deg_t width, deg_t depth, algebra::VectorType vtype) override
    {
        RPY_THROW(std::runtime_error, "set_lie_info is only available for Lie-type channels");
    }
    StreamChannel& add_variant(string variant_label) override
    {
        RPY_THROW(std::runtime_error, "variants are only available for categorical channels");
    }
    StreamChannel& insert_variant(string variant_label) override
    {
        RPY_THROW(std::runtime_error, "variants are only available for categorical channels");
    }
};



}// namespace python
}// namespace rpy

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

    // py::class_<StreamChannel, std::shared_ptr<StreamChannel>> cls(m,
    // "StreamChannel");
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
            switch (item.second->type()) {
                case streams::ChannelType::Increment:
                    PyList_SET_ITEM(
                            plist, i++, PyUnicode_FromString(item.first.c_str())
                    );
                    break;
                case streams::ChannelType::Value:
                    if (item.second->is_lead_lag()) {
                        PyList_SET_ITEM(
                                plist, i++,
                                PyUnicode_FromString(
                                        (item.first
                                         + item.second->label_suffix(0))
                                                .c_str()
                                )
                        );
                        PyList_SET_ITEM(
                                plist, i++,
                                PyUnicode_FromString(
                                        (item.first
                                         + item.second->label_suffix(1))
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
                    auto nvariants = item.second->num_variants();
                    for (dimn_t idx = 0; idx < nvariants; ++idx) {
                        PyList_SET_ITEM(
                                plist, i++,
                                PyUnicode_FromString(
                                        (item.first
                                         + item.second->label_suffix(idx))
                                                .c_str()
                                )
                        );
                    }
                    break;
                }
                case streams::ChannelType::Lie: break;
            }
        }
        RPY_CHECK(i == static_cast<deg_t>(schema->width()));
        return labels;
    });
}
