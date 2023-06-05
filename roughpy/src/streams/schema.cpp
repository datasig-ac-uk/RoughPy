// Copyright (c) 2023 Datasig Developers. All rights reserved.
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


using namespace rpy;
using namespace rpy::python;
using namespace rpy::streams;
using namespace pybind11::literals;

namespace {


class PyChannelItem {
    StreamChannel* ptr;

public:

    PyChannelItem(StreamChannel& item) : ptr(&item)
    {}


};

inline void init_channel_item(py::module_& m) {

    py::enum_<ChannelType>(m, "ChannelType")
        .value("Increment", ChannelType::Increment)
        .value("Value", ChannelType::Value)
        .value("Categorical", ChannelType::Categorical)
        .export_values();

    py::class_<StreamChannel> cls(m, "StreamChannel");

}


}



void rpy::python::init_schema(py::module_ &m) {

    init_channel_item(m);

    py::class_<StreamSchema, std::shared_ptr<StreamSchema>>
        cls(m, "StreamSchema");


    cls.def_static("from_data", &parse_schema_from_data, "data"_a);
    cls.def_static("parse", &parse_schema, "schema"_a);

    cls.def("width", &StreamSchema::width);

    cls.def("insert_increment", [](StreamSchema* schema, string label) {
        return &schema->insert_increment(std::move(label));
    }, "label"_a, py::return_value_policy::reference_internal);

    cls.def("insert_value", [](StreamSchema* schema, string label) {
        return &schema->insert_value(std::move(label));
    }, "label"_a, py::return_value_policy::reference_internal);

    cls.def("insert_categorical", [](StreamSchema* schema, string label) {
        return &schema->insert_categorical(std::move(label));
    }, "label"_a, py::return_value_policy::reference_internal);

    cls.def("get_labels", [](const StreamSchema* schema) {
        py::list labels(schema->width());
        auto* plist = labels.ptr();
        py::ssize_t i = 0;
        for (auto&& item : *schema) {
            switch(item.second.type()) {
                case streams::ChannelType::Increment:
                    PyList_SET_ITEM(plist, i++, PyUnicode_FromString(item.first.c_str()));
                    break;
                case streams::ChannelType::Value:
                    PyList_SET_ITEM(plist, i++, PyUnicode_FromString((item.first + item.second.label_suffix(0)).c_str()));
                    PyList_SET_ITEM(plist, i++, PyUnicode_FromString((item.first + item.second.label_suffix(1)).c_str()));
                    break;
                case streams::ChannelType::Categorical:
                    auto nvariants = item.second.num_variants();
                    for (dimn_t idx=0; idx<nvariants; ++idx) {
                        PyList_SET_ITEM(plist, i++,
                                        PyUnicode_FromString((item.first + item.second.label_suffix(idx)).c_str()));
                    }
                    break;
            }
        }
        return labels;
    });

}


static std::shared_ptr<StreamSchema> parse_schema_from_dict_data(const py::dict& dict_data) {
    std::shared_ptr<StreamSchema> result(new StreamSchema);

    for (auto&& [timestamp, datum] : dict_data) {
        if (py::isinstance<py::tuple>(datum)) {
            auto tpl_datum = py::reinterpret_borrow<py::tuple>(datum);
            auto len = py::len(datum);
            if (len == 2) {
                // Assume increment if datum[1] is not a string, and categorical otherwise
                if (py::isinstance<py::str>(tpl_datum[1])) {
                    auto& item = result->insert_categorical(tpl_datum[0].cast<string>());
                    item.add_variant(tpl_datum[1].cast<string>());
                } else {
                    result->insert_increment(tpl_datum[0].cast<string>());
                }
            } else if (len == 3){
                RPY_CHECK(py::isinstance<py::str>(tpl_datum[1]));
                auto type = tpl_datum[1].cast<string_view>();

                if (type == "increment") {
                    result->insert_increment(tpl_datum[0].cast<string>());
                } else if (type == "value") {
                    result->insert_value(tpl_datum[0].cast<string>());
                } else if (type == "categorical") {
                    auto& cat = result->insert_categorical(tpl_datum[0].cast<string>());
                    cat.add_variant(tpl_datum[1].cast<string>());
                } else {
                    throw py::value_error("unknown type " + string(type));
                }

            } else {
                throw py::value_error("expected a tuple (label, [type,] value)");
            }

        }
    }

    return result;
}
static std::shared_ptr<StreamSchema> parse_schema_from_seq_data(const py::sequence& sequence) {

    std::shared_ptr<StreamSchema> result(new StreamSchema);

    for (auto&& datum : sequence) {
        RPY_CHECK(py::isinstance<py::tuple>(datum));
        auto item = py::reinterpret_borrow<py::tuple>(datum);
        auto len = py::len(item);
        RPY_CHECK(len > 1);
        RPY_CHECK(py::isinstance<py::float_>(item[0]));

        if (len == 2) {
            /// (timestamp, value)
            if (py::isinstance<py::str>(item[1])) {
                auto& cat = result->insert_categorical("");
                cat.add_variant(item[1].cast<string>());
            } else {
                result->insert_increment("");
            }
        } else if (len == 3) {
            /// (timestamp, label, value)
            RPY_CHECK(py::isinstance<py::str>(item[1]));
            if (py::isinstance<py::str>(item[2])) {
                auto& cat = result->insert_categorical(item[1].cast<string>());
                cat.add_variant(item[2].cast<string>());
            } else {
                result->insert_increment(item[1].cast<string>());
            }
        } else if (len == 4) {
            RPY_CHECK(py::isinstance<py::str>(item[1]));
            RPY_CHECK(py::isinstance<py::str>(item[2]));

            auto type = item[2].cast<string>();

            if (type == "increment") {
                result->insert_increment(item[1].cast<string>());
            } else if (type == "value") {
                result->insert_value(item[1].cast<string>());
            } else if (type == "categorical") {
                auto& cat = result->insert_categorical(item[1].cast<string>());
                cat.add_variant(item[3].cast<string>());
            } else {
                throw py::value_error("unknown type " + type);
            }

        } else {
            throw py::value_error("expected tuple (timestamp, [label, [type,]] value)");
        }
    }

    return result;
}


std::shared_ptr<streams::StreamSchema> rpy::python::parse_schema_from_data(const py::object &data) {

    if (py::isinstance<py::dict>(data)) {
        return parse_schema_from_dict_data(py::reinterpret_borrow<py::dict>(data));
    }

    if (py::isinstance<py::sequence>(data)) {
        return parse_schema_from_seq_data(py::reinterpret_borrow<py::sequence>(data));
    }

    throw py::type_error("expected sequential data");
}


void parse_schema_from_dict(StreamSchema& schema, const py::dict& data) {
    throw std::runtime_error("not yet supported");
}

void parse_schema_from_sequence(StreamSchema& schema, const py::sequence& data) {

    for (auto&& item : data) {
        RPY_CHECK(py::isinstance<py::dict>(item));

        auto has_variants = item.contains("variants");
        ChannelType ctype = rpy::streams::ChannelType::Increment;
        if (item.contains("type")) {
            auto type = item["type"].cast<string>();
            if (type == "increment") {
                // Already set
            } else if (type == "value") {
                ctype = rpy::streams::ChannelType::Value;
            } else if (type == "categorical" || has_variants) {
                ctype = rpy::streams::ChannelType::Categorical;
            } else {
                throw py::value_error("unknown type " + type);
            }
        } else if (has_variants) {
            ctype = rpy::streams::ChannelType::Categorical;
        }

        if (has_variants) {
            if (ctype != rpy::streams::ChannelType::Categorical) {
                throw py::value_error("only categorical channels may have variants");
            }

            RPY_CHECK(py::isinstance<py::sequence>(item["variants"]));
        } else {
            if (ctype == rpy::streams::ChannelType::Categorical) {
                throw py::value_error("categorical channels must have a sequence of variants");
            }
        }

        string label;
        if (item.contains("label")) {
            label = item["label"].cast<string>();
        }

        switch (ctype) {
            case rpy::streams::ChannelType::Increment:
                schema.insert_increment(label);
                break;
            case rpy::streams::ChannelType::Value:
                schema.insert_value(label);
                break;
            case rpy::streams::ChannelType::Categorical:
                auto& cat = schema.insert_categorical(label);
                for (auto&& var : item["variants"]) {
                    cat.add_variant(var.cast<string>());
                }
                break;
        }
    }


}


std::shared_ptr<streams::StreamSchema> rpy::python::parse_schema(const py::object &data) {
    std::shared_ptr<StreamSchema> result(new StreamSchema);

    if (py::isinstance<py::dict>(data)) {
        parse_schema_from_dict(*result, py::reinterpret_borrow<py::dict>(data));
    } else if (py::isinstance<py::sequence>(data)) {
        parse_schema_from_sequence(*result, py::reinterpret_borrow<py::sequence>(data));
    } else {
        throw py::type_error("expected dict or sequence");
    }


    return result;
}
