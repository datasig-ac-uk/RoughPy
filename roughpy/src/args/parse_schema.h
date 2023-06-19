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

std::shared_ptr<streams::StreamSchema> parse_schema_from_data(const py::object &data);
std::shared_ptr<streams::StreamSchema> parse_schema(const py::object &data);

}
}



#endif//ROUGHPY_PARSE_SCHEMA_H
