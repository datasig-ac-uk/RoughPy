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

streams::ChannelType python::py_to_channel_type(const py::object &arg) {
    if (py::isinstance<streams::ChannelType>(arg)) {
        return arg.cast<streams::ChannelType>();
    }

    if (py::isinstance<py::str>(arg)) {
        return string_to_channel_type(arg.cast<string>());
    }

    throw py::type_error("no know conversion from " + arg.get_type().attr("__name__").cast<string>() + " to channel type");
}

static void parse_schema_from_dict_data(StreamSchema* result, const py::dict &dict_data) {

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
        } else if (py::isinstance<py::list>(datum)) {

        } else {
            throw py::type_error("expected tuple, list, or dict");
        }
    }

}
static void parse_schema_from_seq_data(StreamSchema* schema, const py::sequence &sequence) {

    for (auto &&datum : sequence) {
        RPY_CHECK(py::isinstance<py::tuple>(datum));
        auto item = py::reinterpret_borrow<py::tuple>(datum);
        auto len = py::len(item);
        RPY_CHECK(len > 1);
        RPY_CHECK(py::isinstance<py::float_>(item[0]));

        if (len == 2) {
            /// (timestamp, value)
            if (py::isinstance<py::str>(item[1])) {
                auto &cat = schema->insert_categorical("");
                cat.add_variant(item[1].cast<string>());
            } else {
                schema->insert_increment("");
            }
        } else if (len == 3) {
            /// (timestamp, label, value)
            RPY_CHECK(py::isinstance<py::str>(item[1]));
            if (py::isinstance<py::str>(item[2])) {
                auto &cat = schema->insert_categorical(item[1].cast<string>());
                cat.add_variant(item[2].cast<string>());
            } else {
                schema->insert_increment(item[1].cast<string>());
            }
        } else if (len == 4) {
            RPY_CHECK(py::isinstance<py::str>(item[1]));
            RPY_CHECK(py::isinstance<py::str>(item[2]));

            auto type = item[2].cast<string>();

            if (type == "increment") {
                schema->insert_increment(item[1].cast<string>());
            } else if (type == "value") {
                schema->insert_value(item[1].cast<string>());
            } else if (type == "categorical") {
                auto &cat = schema->insert_categorical(item[1].cast<string>());
                cat.add_variant(item[3].cast<string>());
            } else {
                throw py::value_error("unknown type " + type);
            }

        } else {
            throw py::value_error("expected tuple (timestamp, [label, [type,]] value)");
        }
    }

}

void parse_schema_from_dict(StreamSchema &schema, const py::dict &data) {
    (void) schema;
    (void) data;
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
            case rpy::streams::ChannelType::Categorical: {
                auto &cat = schema.insert_categorical(label);
                for (auto &&var : item["variants"]) {
                    cat.add_variant(var.cast<string>());
                }
                break;
            }
            case rpy::streams::ChannelType::Lie:
                schema.insert_lie(label);
                break;
        }
    }
}


namespace {

void handle_timestamp_pair(StreamSchema *schema, py::object data);


std::pair<ChannelType, string> get_type_and_variant(StreamSchema* schema, py::object data) {

}

inline void handle_labeled_data(StreamSchema* schema, string label, py::object data) {
    auto [type, variant] = get_type_and_variant(schema, std::move(data));
    auto found = schema->find(label);
    if (found != schema->end()) {
        RPY_CHECK(type == found->second.type());

        if (type == rpy::streams::ChannelType::Categorical) {
            found->second.insert_variant(std::move(variant));
        }
        //TODO: add checks for other data types
    } else {
        switch (type) {
            case ChannelType::Increment:
                schema->insert_increment(std::move(label));
                break;
            case ChannelType::Value:
                schema->insert_value(std::move(label));
                break;
            case ChannelType::Categorical:
                schema->insert_categorical(std::move(label))
                        .add_variant(std::move(variant));
                break;
            case ChannelType::Lie:
                // TODO: Handle Lie case?
                break;
        }
    }
}

/**
 * @brief Handle tuple of (label, *data)
 */
inline void handle_data_tuple(StreamSchema* schema, py::sequence seq) {
    RPY_CHECK(py::len(seq) > 1);

}

/**
 * @brief Handle dict of {label: value, ...}
 *
 */
inline void handle_data_dict(StreamSchema* schema, py::dict data_dict) {

}

void handle_timestamp_pair(StreamSchema *schema, py::object data) {

    if (py::isinstance<py::dict>(data)) {
        auto data_dict = py::reinterpret_steal<py::dict>(data);
        handle_data_dict(schema, data_dict);
    } else if (py::isinstance<py::tuple>(data)) {
        auto data_tuple = py::reinterpret_steal<py::tuple>(data);
        handle_data_tuple(schema, data_tuple);
    } else if (py::isinstance<py::sequence>(data)) {
        auto data_seq = py::reinterpret_steal<py::sequence>(data);

        for (auto&& it_data : data_seq) {
            auto inner = py::reinterpret_borrow<py::object>(it_data);
            handle_timestamp_pair(schema, std::move(inner));
        }
    } else {
        throw py::value_error("expected dict, tuple, or other sequence");
    }

}

inline void handle_dict_stream(StreamSchema* schema, const py::dict& data) {
    for (auto&& [timestamp, tick_value] : data) {
        handle_timestamp_pair(schema, py::reinterpret_borrow<py::object>(tick_value));
    }
}

inline void handle_tuple_sequence(StreamSchema* schema, const py::sequence& data) {
    for (auto&& it_value : data) {
        RPY_CHECK(py::isinstance<py::sequence>(it_value));
        auto inner = py::reinterpret_borrow<py::sequence>(it_value);
        auto len = py::len(inner);

        RPY_CHECK(len > 1 && len <= 4);
        auto right = inner[py::slice(1, {}, {})];
        handle_timestamp_pair(schema, right);
    }
}


}
void python::parse_into_schema(std::shared_ptr<streams::StreamSchema> schema, const py::object &data) {

    if (py::isinstance<py::dict>(data)) {
        parse_schema_from_dict(*schema, py::reinterpret_borrow<py::dict>(data));
    } else if (py::isinstance<py::sequence>(data)) {
        parse_schema_from_sequence(*schema, py::reinterpret_borrow<py::sequence>(data));
    } else {
        throw py::type_error("expected dict or sequence");
    }
}

void python::parse_data_into_schema(std::shared_ptr<streams::StreamSchema> schema, const py::object &data) {
    if (py::isinstance<py::dict>(data)) {
        parse_schema_from_dict_data(schema.get(), py::reinterpret_borrow<py::dict>(data));
    } else if (py::isinstance<py::sequence>(data)) {
        parse_schema_from_seq_data(schema.get(), py::reinterpret_borrow<py::sequence>(data));
    } else {
        throw py::type_error("expected sequential data");
    }
}

std::shared_ptr<streams::StreamSchema> rpy::python::parse_schema(const py::object &data) {
    auto result = std::make_shared<streams::StreamSchema>();
    parse_into_schema(result, data);
    return result;
}

std::shared_ptr<streams::StreamSchema> rpy::python::parse_schema_from_data(const py::object &data) {
    auto schema = std::make_shared<streams::StreamSchema>();
    parse_data_into_schema(schema, data);
    return schema;
}
