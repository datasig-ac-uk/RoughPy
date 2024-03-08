//
// Created by sam on 2/26/24.
//

#ifndef ROUGHPY_PARSE_DATA_ARGUMENT_H
#define ROUGHPY_PARSE_DATA_ARGUMENT_H

#include "roughpy_module.h"

#include "scalars/scalars.h"
#include <roughpy/core/types.h>
#include <roughpy/platform/devices/core.h>
#include <roughpy/scalars/scalars_fwd.h>

#include <boost/container/small_vector.hpp>

#include <vector>

namespace rpy {
namespace python {

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

struct LeafData {

    LeafData(
            boost::container::small_vector<idimn_t, 1>&& shape,
            scalars::KeyScalarArray&& data,
            py::object&& owningObject,
            dimn_t size,
            devices::TypeInfo originTypeInfo,
            LeafType leafType,
            ValueType valueType
    )
        : shape(std::move(shape)),
          data(std::move(data)),
          owning_object(std::move(owningObject)),
          size(size),
          origin_type_info(originTypeInfo),
          leaf_type(leafType),
          value_type(valueType)
    {}

    boost::container::small_vector<idimn_t, 1> shape;
    scalars::KeyScalarArray data;
    py::object owning_object;
    dimn_t size;
    devices::TypeInfo origin_type_info;
    LeafType leaf_type;
    ValueType value_type;
};

class ParsedData : public std::vector<LeafData>
{
    std::vector<param_t> m_indices;

public:
    const std::vector<param_t>& indices() const noexcept { return m_indices; }

    void fill_ks_stream(scalars::KeyScalarStream& ks_stream);

    void fill_ks_buffer(scalars::KeyScalarArray& ks_array);
};

// struct RPY_NO_EXPORT AlternativeKeyType {
//     py::handle py_key_type;
//     std::function<key_type(py::handle)> converter;
// };

struct RPY_NO_EXPORT DataArgOptions {
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

RPY_NO_DISCARD ParsedData
parse_data_argument(py::handle arg, DataArgOptions& options);

}// namespace python
}// namespace rpy

#endif// ROUGHPY_PARSE_DATA_ARGUMENT_H
