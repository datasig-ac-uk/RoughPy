//
// Created by sam on 18/06/23.
//

#ifndef ROUGHPY_PARSE_SCHEMA_H
#define ROUGHPY_PARSE_SCHEMA_H

#include "roughpy_module.h"
#include <roughpy/streams/schema.h>

namespace rpy {
namespace python {

streams::ChannelType string_to_channel_type(string channel_str);

streams::ChannelType py_to_channel_type(const py::object& arg);

void parse_data_into_schema(
        std::shared_ptr<streams::StreamSchema> schema, const py::object& data
);
void parse_into_schema(
        std::shared_ptr<streams::StreamSchema> schema, const py::object& data
);

std::shared_ptr<streams::StreamSchema>
parse_schema_from_data(const py::object& data);
std::shared_ptr<streams::StreamSchema> parse_schema(const py::object& data);

}// namespace python
}// namespace rpy

#endif// ROUGHPY_PARSE_SCHEMA_H
