//
// Created by sam on 08/08/23.
//

#ifndef ROUGHPY_PARSE_KEY_SCALAR_STREAM_H
#define ROUGHPY_PARSE_KEY_SCALAR_STREAM_H

#include "roughpy_module.h"

#include <roughpy/scalars/key_scalar_stream.h>
#include <roughpy/scalars/owned_scalar_array.h>

#include "scalar_type.h"
#include "scalars.h"

namespace rpy {
namespace python {

struct ParsedKeyScalarStream {
    /// Parsed key-scalar stream
    scalars::KeyScalarStream data_stream;

    /// Buffer holding key/scalar data if a copy had to be made
    scalars::KeyScalarArray data_buffer;
};

string dl_to_type_id(const DLDataType& dtype, const DLDevice& device);

void parse_key_scalar_stream(
        ParsedKeyScalarStream& result, const py::object& data,
        PyToBufferOptions& options
);

}// namespace python
}// namespace rpy

#endif// ROUGHPY_PARSE_KEY_SCALAR_STREAM_H
