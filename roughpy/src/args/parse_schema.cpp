//
// Created by sam on 18/06/23.
//

#include "parse_schema.h"
#include <cctype>

using namespace rpy;
using namespace rpy::python;
using namespace rpy::streams;

streams::ChannelType rpy::python::string_to_channel_type(string channel_str)
{
    std::transform(
            channel_str.begin(), channel_str.end(), channel_str.begin(),
            [](unsigned char c) { return std::tolower(c); }
    );

    if (channel_str == "increment") {
        return streams::ChannelType::Increment;
    } else if (channel_str == "value") {
        return streams::ChannelType::Value;
    } else if (channel_str == "categorical") {
        return streams::ChannelType::Categorical;
    } else if (channel_str == "lie") {
        return streams::ChannelType::Lie;
    } else {
        RPY_THROW(py::value_error,"expected increment, value, categorical, or lie "
                              "for channel type");
    }
}

streams::ChannelType python::py_to_channel_type(const py::object& arg)
{
    if (py::isinstance<streams::ChannelType>(arg)) {
        return arg.cast<streams::ChannelType>();
    }

    if (py::isinstance<py::str>(arg)) {
        return string_to_channel_type(arg.cast<string>());
    }

    RPY_THROW(py::type_error,
            "no know conversion from "
            + arg.get_type().attr("__name__").cast<string>()
            + " to channel type"
    );
}

namespace {

inline void insert_increment_to_schema(
        StreamSchema* schema, string label, py::dict RPY_UNUSED_VAR options
)
{
    // No options for increment types at the moment
    auto& channel RPY_UNUSED_VAR = schema->insert_increment(std::move(label));
}
inline void
insert_value_to_schema(StreamSchema* schema, string label, py::dict options)
{
    auto& channel = schema->insert_value(std::move(label));

    if (options.contains("lead_lag")) {
        channel.set_lead_lag(options["lead_lag"].cast<bool>());
    }
}
inline void insert_categorical_to_schema(
        StreamSchema* schema, string label, py::dict options
)
{
    auto& channel = schema->insert_categorical(std::move(label));

    if (options.contains("categories")) {
        for (auto&& cat : options["categories"]) {
            channel.insert_variant(cat.cast<string>());
        }
    }
}
inline void insert_lie_to_schema(
        StreamSchema* schema, string label, py::dict RPY_UNUSED_VAR options
)
{
    auto& channel RPY_UNUSED_VAR = schema->insert_lie(std::move(label));
}

inline void insert_item_to_schema(
        StreamSchema* schema, string label, ChannelType type, py::dict options
)
{
    switch (type) {
        case ChannelType::Increment:
            insert_increment_to_schema(
                    schema, std::move(label), std::move(options)
            );
            break;
        case ChannelType::Value:
            insert_value_to_schema(
                    schema, std::move(label), std::move(options)
            );
            break;
        case ChannelType::Categorical:
            insert_categorical_to_schema(
                    schema, std::move(label), std::move(options)
            );
            break;
        case ChannelType::Lie:
            insert_lie_to_schema(schema, std::move(label), std::move(options));
            break;
    }
}

void handle_seq_item(StreamSchema* schema, string label, py::sequence data)
{

    auto len = py::len(data);

    RPY_CHECK(len > 1);
    auto type = py_to_channel_type(data[0]);

    if (len == 1) {
        insert_item_to_schema(schema, std::move(label), type, py::dict());
    } else if (len == 2) {
        if (!py::isinstance<py::dict>(data[1])) {
            RPY_THROW(py::type_error, "options must be a dictionary if provided");
        }
        insert_item_to_schema(
                schema, std::move(label), type,
                py::reinterpret_borrow<py::dict>(data[1])
        );
    } else {
        RPY_THROW(py::value_error, "expected tuple , type [, options])");
    }
}

void handle_dict_item(StreamSchema* schema, string label, py::dict data)
{
    /*
     * The item contains a mapping of key -> values, which must include "type",
     * along with any other options to be appended to the schema
     */
    if (!data.contains("type")) {
        RPY_THROW(py::value_error, "dict items must contain \"type\"");
    }

    auto type = py_to_channel_type(data["type"]);
    // Everything remaining should be an option to be passed
    auto data_copy = py::reinterpret_steal<py::dict>(PyDict_Copy(data.ptr()));
    PyDict_DelItemString(data_copy.ptr(), "type");

    insert_item_to_schema(schema, std::move(label), type, std::move(data_copy));
}
void handle_dict_item_no_label(StreamSchema* schema, py::dict data)
{
    if (!data.contains("label")) {
        RPY_THROW(py::value_error, "dict items in a schema must contain \"label\"");
    }
    auto data_copy = py::reinterpret_steal<py::dict>(PyDict_Copy(data.ptr()));

    auto label = data_copy["label"].cast<string>();
    PyDict_DelItemString(data_copy.ptr(), "label");

    handle_dict_item(schema, std::move(label), std::move(data_copy));
}

void handle_seq_item_no_label(StreamSchema* schema, py::sequence data)
{
    RPY_CHECK(py::len(data) > 1);
    handle_seq_item(schema, data[0].cast<string>(), data[py::slice(1, {}, {})]);
}

void handle_dict_schema(StreamSchema* schema, py::dict data)
{
    for (auto&& [label, item] : data) {
        if (py::isinstance<py::dict>(item)) {
            handle_dict_item(
                    schema, label.cast<string>(),
                    py::reinterpret_borrow<py::dict>(item)
            );
        } else if (py::isinstance<py::sequence>(item)) {
            handle_seq_item(
                    schema, label.cast<string>(),
                    py::reinterpret_borrow<py::sequence>(item)
            );
        } else {
            RPY_THROW(py::type_error, "unsupported type in schema specification");
        }
    }
}

void handle_seq_schema(StreamSchema* schema, py::sequence data)
{

    for (auto&& item : data) {
        if (py::isinstance<py::dict>(item)) {
            handle_dict_item_no_label(
                    schema, py::reinterpret_borrow<py::dict>(item)
            );
        } else if (py::isinstance<py::sequence>(item)) {
            handle_seq_item_no_label(
                    schema, py::reinterpret_borrow<py::sequence>(item)
            );
        } else {
            RPY_THROW(py::type_error, "unsupported type in schema specification");
        }
    }
}

//void handle_timestamp_pair(StreamSchema* schema, py::object data);
//
//RPY_NO_DISCARD
//pair<ChannelType, string>
//get_type_and_variant(StreamSchema* schema, py::object data)
//{
//    if (py::isinstance<py::dict>(data)) {
//        auto data_dict = py::reinterpret_borrow<py::dict>(data);
//        RPY_CHECK(data_dict.contains("data"));
//
//        if (data_dict.contains("type")) {
//            auto type = py_to_channel_type(data_dict["type"]);
//            if (type == ChannelType::Categorical) {
//                return {type, data_dict["data"].cast<string>()};
//            }
//            return {type, {}};
//        }
//
//        return get_type_and_variant(schema, data_dict["data"]);
//    }
//
//    if (py::isinstance<py::tuple>(data)) {
//        auto data_tuple = py::reinterpret_borrow<py::tuple>(data);
//        if (data_tuple.size() == 1) {
//            return get_type_and_variant(
//                    schema, py::reinterpret_borrow<py::object>(data_tuple[0])
//            );
//        }
//
//        // data is (type, data, ...)
//        auto type = py_to_channel_type(data_tuple[0]);
//        if (type == ChannelType::Categorical) {
//            return {type, data_tuple[1].cast<string>()};
//        }
//
//        return {type, {}};
//    }
//
//    // The data is a plain value, infer from type
//    if (py::isinstance<py::str>(data)) {
//        return {ChannelType::Categorical, data.cast<string>()};
//    }
//
//    // TODO: IN the future, use the schema to determine what should be
//    // returned here, but for now hard-code increment
//    return {ChannelType::Increment, {}};
//}
//
//inline void
//handle_labeled_data(StreamSchema* schema, string label, py::object data)
//{
//    auto [type, variant] = get_type_and_variant(schema, std::move(data));
//    auto found = schema->find(label);
//    if (found != schema->end()) {
//        RPY_CHECK(type == found->second.type());
//
//        if (type == ChannelType::Categorical) {
//            found->second.insert_variant(std::move(variant));
//        }
//        // TODO: add checks for other data types
//    } else {
//        switch (type) {
//            case ChannelType::Increment:
//                schema->insert_increment(std::move(label));
//                break;
//            case ChannelType::Value:
//                schema->insert_value(std::move(label));
//                break;
//            case ChannelType::Categorical:
//                schema->insert_categorical(std::move(label))
//                        .add_variant(std::move(variant));
//                break;
//            case ChannelType::Lie:
//                // TODO: Handle Lie case?
//                break;
//        }
//    }
//}
//
///**
// * @brief Handle tuple of (label, *data)
// */
//inline void handle_data_tuple(StreamSchema* schema, const py::sequence& seq)
//{
//    auto length = py::len(seq);
//    RPY_CHECK(length > 1);
//
//    auto label = seq[0].cast<string>();
//    handle_labeled_data(schema, std::move(label), seq[py::slice(1, {}, {})]);
//}
//
///**
// * @brief Handle dict of {label: value, ...}
// *
// */
//inline void handle_data_dict(StreamSchema* schema, const py::dict& data_dict)
//{
//    //    py::print(data_dict);
//    //    RPY_CHECK(data_dict.contains("label"));
//    //    RPY_CHECK(data_dict.contains("data"));
//    //    auto label = data_dict["label"].cast<string>();
//    //
//    //    if (data_dict.contains("type")) {
//    //        auto to_pass = py::make_tuple(data_dict["type"],
//    //        data_dict["data"]); handle_labeled_data(schema, std::move(label),
//    //        std::move(to_pass));
//    //    } else {
//    //        auto to_pass =
//    //        py::reinterpret_borrow<py::object>(data_dict["data"]);
//    //        handle_labeled_data(schema, std::move(label), std::move(to_pass));
//    //    }
//    for (auto&& [label, value] : data_dict) {
//        auto true_label = label.cast<string>();
//        handle_labeled_data(
//                schema, std::move(true_label),
//                py::reinterpret_borrow<py::object>(value)
//        );
//    }
//}
//
//void handle_timestamp_pair(StreamSchema* schema, py::object data)
//{
//    if (py::isinstance<py::dict>(data)) {
//        auto data_dict = py::reinterpret_borrow<py::dict>(data);
//        handle_data_dict(schema, data_dict);
//    } else if (py::isinstance<py::tuple>(data)) {
//        auto data_tuple = py::reinterpret_borrow<py::tuple>(data);
//        handle_data_tuple(schema, data_tuple);
//    } else if (py::isinstance<py::sequence>(data)) {
//        auto data_seq = py::reinterpret_borrow<py::sequence>(data);
//
//        for (auto&& it_data : data_seq) {
//            auto inner = py::reinterpret_borrow<py::object>(it_data);
//            handle_timestamp_pair(schema, std::move(inner));
//        }
//    } else {
//        RPY_THROW(py::value_error, "expected dict, tuple, or other sequence");
//    }
//}
//
//inline void handle_dict_stream(StreamSchema* schema, const py::dict& data)
//{
//    for (auto&& [timestamp, tick_value] : data) {
//        handle_timestamp_pair(
//                schema, py::reinterpret_borrow<py::object>(tick_value)
//        );
//    }
//}
//
//inline void
//handle_tuple_sequence(StreamSchema* schema, const py::sequence& data)
//{
//    for (auto&& it_value : data) {
//        RPY_CHECK(py::isinstance<py::sequence>(it_value));
//        auto inner = py::reinterpret_borrow<py::sequence>(it_value);
//        auto len = py::len(inner);
//
//        RPY_CHECK(len > 1);
//        if (len == 2) {
//            handle_timestamp_pair(schema, inner[1]);
//        } else if (len <= 4) {
//            auto right = inner[py::slice(1, {}, {})];
//            handle_timestamp_pair(schema, right);
//        } else {
//            RPY_THROW(py::value_error,"expected tuple with no more than 4 elements"
//            );
//        }
//    }
//}

}// namespace

void python::parse_into_schema(
        std::shared_ptr<streams::StreamSchema> schema, const py::object& data
)
{

    if (py::isinstance<py::dict>(data)) {
        handle_dict_schema(
                schema.get(), py::reinterpret_borrow<py::dict>(data)
        );
    } else if (py::isinstance<py::sequence>(data)) {
        handle_seq_schema(
                schema.get(), py::reinterpret_borrow<py::sequence>(data)
        );
    } else {
        RPY_THROW(py::type_error, "expected dict or sequence");
    }
}

// void python::parse_data_into_schema(std::shared_ptr<streams::StreamSchema>
// schema, const py::object &data) {
//     if (py::isinstance<py::dict>(data)) {
////        parse_schema_from_dict_data(schema.get(),
/// py::reinterpret_borrow<py::dict>(data));
//        handle_dict_stream(schema.get(),
//        py::reinterpret_borrow<py::dict>(data));
//    } else if (py::isinstance<py::sequence>(data)) {
////        parse_schema_from_seq_data(schema.get(),
/// py::reinterpret_borrow<py::sequence>(data));
//        handle_tuple_sequence(schema.get(),
//        py::reinterpret_borrow<py::sequence>(data));
//    } else {
//        RPY_THROW(py::type_error, "expected sequential data");
//    }
//}
//
std::shared_ptr<streams::StreamSchema>
rpy::python::parse_schema(const py::object& data)
{
    auto result = std::make_shared<streams::StreamSchema>();
    parse_into_schema(result, data);
    return result;
}

// std::shared_ptr<streams::StreamSchema>
// rpy::python::parse_schema_from_data(const py::object &data) {
//     auto schema = std::make_shared<streams::StreamSchema>();
//     parse_data_into_schema(schema, data);
//     return schema;
// }
