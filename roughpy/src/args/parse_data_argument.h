//
// Created by sam on 2/26/24.
//

#ifndef ROUGHPY_PARSE_DATA_ARGUMENT_H
#define ROUGHPY_PARSE_DATA_ARGUMENT_H

#include "roughpy_module.h"
#include <boost/container/small_vector.hpp>

namespace rpy {
namespace python {

devices::TypeInfo py_type_to_type_info(py::handle pytype);
devices::TypeInfo py_buffer_to_device_info(const py::buffer_info& info);

enum class LeafType : uint8_t
{
    Scalar,
    KeyScalar,
    Lie,
    DLTensor,
    Buffer,
    Dict,
    Sequence
};
enum class ValueType : uint8_t
{
    Value,
    KeyValue
};

struct LeafItem {
    boost::container::small_vector<idimn_t, 1> shape;
    py::object object;
    dimn_t size;
    devices::TypeInfo scalar_info;
    LeafType leaf_type;
    ValueType value_type;
};

struct RPY_NO_EXPORT AlternativeKeyType {
    py::handle py_key_type;
    std::function<key_type(py::handle)> converter;
};

struct RPY_NO_EXPORT DataArgOptions {
    std::vector<LeafItem> leaves;
    std::vector<param_t> indices;
    const scalars::ScalarType* scalar_type = nullptr;
    algebra::context_pointer context = nullptr;
    AlternativeKeyType* alternative_key = nullptr;
    dimn_t max_nested = 0;
    bool allow_none = true;
    bool allow_timestamped = false;
    bool allow_scalar = true;
    bool allow_kv_pair = true;
    bool allow_lie_arg = true;
    bool check_imported_scalars = true;
    bool is_ragged = false;
};

RPY_NO_DISCARD scalars::KeyScalarArray
parse_data_argument(py::handle arg, DataArgOptions& options);

}// namespace python
}// namespace rpy

#endif// ROUGHPY_PARSE_DATA_ARGUMENT_H
