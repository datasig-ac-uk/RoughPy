//
// Created by sam on 18/06/23.
//

#include "parse_schema.h"
#include <cctype>

using namespace rpy;
using namespace rpy::python;
using namespace rpy::streams;

streams::ChannelType rpy::python::string_to_channel_type(string channel_str) {
    std::transform(channel_str.begin(), channel_str.end(), channel_str.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (channel_str == "increment") {
        return streams::ChannelType::Increment;
    } else if (channel_str == "value") {
        return streams::ChannelType::Value;
    } else if (channel_str == "categorical") {
        return streams::ChannelType::Categorical;
    } else if (channel_str == "lie") {
        return streams::ChannelType::Lie;
    } else {
        throw py::value_error("expected increment, value, categorical, or lie for channel type");
    }
}

streams::ChannelType py_to_channel_type(const py::object &arg) {
    if (py::isinstance<streams::ChannelType>(arg)) {
        return arg.cast<streams::ChannelType>();
    }

    if (py::isinstance<py::str>(arg)) {
        return string_to_channel_type(arg.cast<string>());
    }

    throw py::type_error("no know conversion from " + arg.get_type().attr("__name__").cast<string>() + " to channel type");
}

static std::shared_ptr<StreamSchema> parse_schema_from_dict_data(const py::dict &dict_data) {
    std::shared_ptr<StreamSchema> result(new StreamSchema);

    for (auto &&[timestamp, datum] : dict_data) {
        if (py::isinstance<py::tuple>(datum)) {
            auto tpl_datum = py::reinterpret_borrow<py::tuple>(datum);
            auto len = py::len(datum);
            if (len == 2) {
                // Assume increment if datum[1] is not a string, and categorical otherwise
                if (py::isinstance<py::str>(tpl_datum[1])) {
                    auto &item = result->insert_categorical(tpl_datum[0].cast<string>());
                    item.add_variant(tpl_datum[1].cast<string>());
                } else {
                    result->insert_increment(tpl_datum[0].cast<string>());
                }
            } else if (len == 3) {
                RPY_CHECK(py::isinstance<py::str>(tpl_datum[1]));
                auto type = tpl_datum[1].cast<string_view>();

                if (type == "increment") {
                    result->insert_increment(tpl_datum[0].cast<string>());
                } else if (type == "value") {
                    result->insert_value(tpl_datum[0].cast<string>());
                } else if (type == "categorical") {
                    auto &cat = result->insert_categorical(tpl_datum[0].cast<string>());
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
static std::shared_ptr<StreamSchema> parse_schema_from_seq_data(const py::sequence &sequence) {

    std::shared_ptr<StreamSchema> result(new StreamSchema);

    for (auto &&datum : sequence) {
        RPY_CHECK(py::isinstance<py::tuple>(datum));
        auto item = py::reinterpret_borrow<py::tuple>(datum);
        auto len = py::len(item);
        RPY_CHECK(len > 1);
        RPY_CHECK(py::isinstance<py::float_>(item[0]));

        if (len == 2) {
            /// (timestamp, value)
            if (py::isinstance<py::str>(item[1])) {
                auto &cat = result->insert_categorical("");
                cat.add_variant(item[1].cast<string>());
            } else {
                result->insert_increment("");
            }
        } else if (len == 3) {
            /// (timestamp, label, value)
            RPY_CHECK(py::isinstance<py::str>(item[1]));
            if (py::isinstance<py::str>(item[2])) {
                auto &cat = result->insert_categorical(item[1].cast<string>());
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
                auto &cat = result->insert_categorical(item[1].cast<string>());
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

void parse_schema_from_dict(StreamSchema &schema, const py::dict &data) {
    throw std::runtime_error("not yet supported");
}

void parse_schema_from_sequence(StreamSchema &schema, const py::sequence &data) {

    for (auto &&item : data) {
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
                auto &cat = schema.insert_categorical(label);
                for (auto &&var : item["variants"]) {
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
