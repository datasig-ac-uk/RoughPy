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

RPY_NO_DISCARD
pair<ChannelType, string> get_type_and_variant(StreamSchema* schema, py::object data) {
    if (py::isinstance<py::dict>(data)) {
        auto data_dict = py::reinterpret_borrow<py::dict>(data);
        RPY_CHECK(data_dict.contains("data"));

        if (data_dict.contains("type")) {
            auto type = py_to_channel_type(data_dict["type"]);
            if (type == ChannelType::Categorical) {
                return {type, data_dict["data"].cast<string>()};
            }
            return {type, {}};
        }

        return get_type_and_variant(schema, data_dict["data"]);
    }

    if (py::isinstance<py::tuple>(data)) {
        auto data_tuple = py::reinterpret_borrow<py::tuple>(data);
        if (data_tuple.size() == 1) {
            return get_type_and_variant(schema, py::reinterpret_borrow<py::object>(data_tuple[0]));
        }

        // data is (type, data, ...)
        auto type = py_to_channel_type(data_tuple[0]);
        if (type == ChannelType::Categorical) {
            return {type, data_tuple[1].cast<string>()};
        }

        return {type, {}};
    }

    // The data is a plain value, infer from type
    if (py::isinstance<py::str>(data)) {
        return { ChannelType::Categorical, data.cast<string>() };
    }

    // TODO: IN the future, use the schema to determine what should be
    // returned here, but for now hard-code increment
    return { ChannelType::Increment, {} };
}

inline void handle_labeled_data(StreamSchema* schema, string label, py::object data) {
    auto [type, variant] = get_type_and_variant(schema, std::move(data));
    auto found = schema->find(label);
    if (found != schema->end()) {
        RPY_CHECK(type == found->second.type());

        if (type == ChannelType::Categorical) {
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
inline void handle_data_tuple(StreamSchema* schema, const py::sequence& seq) {
    auto length = py::len(seq);
    RPY_CHECK(length > 1);

    auto label = seq[0].cast<string>();
    handle_labeled_data(schema, std::move(label), seq[py::slice(1, {}, {})]);
}

/**
 * @brief Handle dict of {label: value, ...}
 *
 */
inline void handle_data_dict(StreamSchema* schema, const py::dict& data_dict) {
//    py::print(data_dict);
//    RPY_CHECK(data_dict.contains("label"));
//    RPY_CHECK(data_dict.contains("data"));
//    auto label = data_dict["label"].cast<string>();
//
//    if (data_dict.contains("type")) {
//        auto to_pass = py::make_tuple(data_dict["type"], data_dict["data"]);
//        handle_labeled_data(schema, std::move(label), std::move(to_pass));
//    } else {
//        auto to_pass = py::reinterpret_borrow<py::object>(data_dict["data"]);
//        handle_labeled_data(schema, std::move(label), std::move(to_pass));
//    }
    for (auto&& [label, value] : data_dict) {
        auto true_label = label.cast<string>();
        handle_labeled_data(schema,
                            std::move(true_label),
                            py::reinterpret_borrow<py::object>(value));
    }

}

void handle_timestamp_pair(StreamSchema *schema, py::object data) {
    if (py::isinstance<py::dict>(data)) {
        auto data_dict = py::reinterpret_borrow<py::dict>(data);
        handle_data_dict(schema, data_dict);
    } else if (py::isinstance<py::tuple>(data)) {
        auto data_tuple = py::reinterpret_borrow<py::tuple>(data);
        handle_data_tuple(schema, data_tuple);
    } else if (py::isinstance<py::sequence>(data)) {
        auto data_seq = py::reinterpret_borrow<py::sequence>(data);

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

        RPY_CHECK(len > 1);
        if (len == 2) {
            handle_timestamp_pair(schema, inner[1]);
        } else if (len <= 4) {
            auto right = inner[py::slice(1, {}, {})];
            handle_timestamp_pair(schema, right);
        } else {
            throw py::value_error("expected tuple with no more than 4 elements");
        }
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
//        parse_schema_from_dict_data(schema.get(), py::reinterpret_borrow<py::dict>(data));
        handle_dict_stream(schema.get(), py::reinterpret_borrow<py::dict>(data));
    } else if (py::isinstance<py::sequence>(data)) {
//        parse_schema_from_seq_data(schema.get(), py::reinterpret_borrow<py::sequence>(data));
        handle_tuple_sequence(schema.get(), py::reinterpret_borrow<py::sequence>(data));
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
