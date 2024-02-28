//
// Created by sam on 2/26/24.
//

#include "parse_data_argument.h"

#include "dlpack.h"
#include "dlpack_helpers.h"
#include "numpy.h"

#include "scalars/r_py_polynomial.h"
#include "scalars/scalar.h"
#include "scalars/scalar_type.h"

#include <roughpy/algebra/lie.h>
#include <roughpy/scalars/key_scalar_array.h>
#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_type.h>

#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <boost/container/small_vector.hpp>

#include <algorithm>
#include <functional>

using namespace rpy;
using namespace rpy::python;

scalars::Scalar py_to_scalar(const scalars::ScalarType* type, py::handle arg);

namespace {

key_type py_to_key(py::handle arg);

class ConversionManager
{
    scalars::KeyScalarArray m_data;
    DataArgOptions& m_options;
    dimn_t m_offset = 0;

public:
    explicit ConversionManager(DataArgOptions& options)
        : m_data(options.scalar_type),
          m_options(options)
    {}

    scalars::KeyScalarArray take() && { return m_data; }

private:
    RPY_NO_DISCARD bool is_sparse() const noexcept { return m_data.has_keys(); }

    RPY_NO_DISCARD key_type compute_key() const noexcept;

    LeafItem& add_leaf(py::handle node, LeafType type);

    void push(scalars::Scalar scalar);
    void push(key_type key, scalars::Scalar scalar);
    void push(const algebra::Lie& lie);
    void push(py::buffer_info buf_info);
    void push(DLTensor* dl_tensor);

    void compute_size_and_allocate();

    void handle_scalar(py::handle value);
    void handle_key_scalar(py::handle value);
    void handle_lie(py::handle value);
    void handle_dltensor(py::handle value);
    void handle_buffer(py::handle value);
    void handle_dict(py::handle value);
    void handle_sequence(py::handle value);
    void do_conversion();

    void check_dl_size(py::capsule dlcap, deg_t depth);
    void check_buffer_size(py::buffer buffer, deg_t depth);

    void check_size_and_type_recurse(py::handle node, deg_t depth);
    void check_size_and_type(py::handle arg);

    RPY_NO_DISCARD const scalars::ScalarType* compute_scalar_type() const;

public:
    void parse_argument(py::handle arg);
};

bool is_pyscalar(py::handle arg) { return false; }

bool is_pykv_pair(py::handle arg, DataArgOptions& options) { return false; }

void check_and_set_dtype(DataArgOptions& options, py::handle arg) {}

void update_dtype_and_allocate(
        scalars::KeyScalarArray& result,
        DataArgOptions& options,
        idimn_t no_values,
        idimn_t no_keys
)
{}

}// namespace

scalars::KeyScalarArray python::parse_data_argument(
        py::handle arg,
        rpy::python::DataArgOptions& options
)
{}

key_type ConversionManager::compute_key() const noexcept { return m_offset; }

LeafItem& ConversionManager::add_leaf(py::handle node, LeafType type)
{
    m_options.leaves.push_back(
            {{}, py::reinterpret_borrow<py::object>(node), {}, type}
    );
    return m_options.leaves.back();
}

void ConversionManager::push(scalars::Scalar value)
{
    RPY_DBG_ASSERT(m_offset < m_data.size());
    m_data[m_offset] = value;
    ++m_offset;
}
void ConversionManager::push(key_type key, scalars::Scalar value)
{
    RPY_DBG_ASSERT(m_data.has_keys() && m_offset < m_data.size());
    m_data.keys()[m_offset] = key;
    push(std::move(value));
}
void ConversionManager::push(const algebra::Lie& lie_data)
{
    RPY_DBG_ASSERT(m_offset < m_data.size());
    if ((lie_data.storage_type() == algebra::VectorType::Dense)
        && !is_sparse()) {
        const auto& dense_data = lie_data.dense_data();

        m_offset += dense_data->size();
    } else {
        for (auto item : lie_data) { push(item.key(), item.value()); }
    }
}
void ConversionManager::push(DLTensor* tensor) {}
void ConversionManager::push(py::buffer_info info) {}

void ConversionManager::compute_size_and_allocate()
{
    RPY_CHECK(!m_leaves.empty());

    bool make_sparse = false;
    idimn_t size = 0;
    for (const auto& leaf : m_options.leaves) {
        idimn_t tmp = 1;
        for (const auto& dim_size : leaf.shape) { tmp *= dim_size; }
        size += tmp;
        make_sparse |= leaf.value_type == ValueType::KeyValue;
    }

    RPY_DBG_ASSERT(size > 0);

    m_data.allocate_scalars(size);
    if (make_sparse) { m_data.allocate_keys(size); }
    RPY_DBG_ASSERT(m_offset == 0);
}

void ConversionManager::handle_scalar(py::handle value)
{
    if (is_sparse()) {
        push(compute_key(), py_to_scalar(m_options.scalar_type, value));
    } else {
        push(py_to_scalar(m_options.scalar_type, value));
    }
}

void ConversionManager::handle_key_scalar(py::handle value)
{
    RPY_DBG_ASSERT(is_sparse());
    RPY_DBG_ASSERT(py::isinstance<py::tuple>(value) && py::len(value) == 2);
    push(py_to_key(value[py::int_(0)]),
         py_to_scalar(m_options.scalar_type, value[py::int_(1)]));
}

void ConversionManager::handle_lie(py::handle value)
{
    push(value.cast<const algebra::Lie&>());
}

void ConversionManager::handle_dltensor(py::handle value)
{
    auto dlpack_capsule = py::reinterpret_borrow<py::capsule>(
            py::getattr(value, "__dlpack__")()
    );
    auto* managed_tensor = dlpack_capsule.get_pointer<DLManagedTensor>();
    push(&managed_tensor->dl_tensor);
}

void ConversionManager::handle_buffer(py::handle value)
{
    push(py::reinterpret_borrow<py::buffer>(value).request());
}

void ConversionManager::handle_dict(py::handle value)
{
    for (auto pair : value) { handle_key_scalar(pair); }
}

void ConversionManager::handle_sequence(py::handle value)
{
    for (auto scalar : value) { handle_scalar(scalar); }
}

void ConversionManager::do_conversion()
{
    compute_size_and_allocate();
    for (const auto& leaf : m_options.leaves) {
        switch (leaf.leaf_type) {
            case LeafType::Scalar: handle_scalar(leaf.object); break;
            case LeafType::KeyScalar: handle_key_scalar(leaf.object); break;
            case LeafType::Lie: handle_lie(leaf.object); break;
            case LeafType::DLTensor: handle_dltensor(leaf.object); break;
            case LeafType::Buffer: handle_buffer(leaf.object); break;
            case LeafType::Dict: handle_dict(leaf.object); break;
            case LeafType::Sequence: handle_sequence(leaf.object); break;
        }
    }
}

void ConversionManager::check_dl_size(py::capsule dlcap, deg_t depth)
{
    auto* managed_tensor = unpack_dl_capsule(dlcap);
    auto& tensor = managed_tensor->dl_tensor;

    depth += tensor.ndim;
    RPY_CHECK(depth <= m_options.max_nested);

    auto& leaf = add_leaf(std::move(dlcap), LeafType::DLTensor);

    leaf.shape.assign(tensor.shape, tensor.shape + tensor.ndim);

    leaf.value_type = ValueType::Value;
    leaf.scalar_info = convert_from_dl_datatype(tensor.dtype);
}

void ConversionManager::check_buffer_size(py::buffer buffer, deg_t depth)
{
    const auto info = buffer.request();

    depth += info.ndim;
    RPY_CHECK(depth <= m_options.max_nested);

    auto& leaf = add_leaf(std::move(buffer), LeafType::Buffer);

    leaf.value_type = ValueType::Value;
    leaf.shape.reserve(info.ndim);
    for (const auto& dim : info.shape) { leaf.shape.push_back(dim); }

    leaf.scalar_info = py_buffer_to_device_info(info);
}

void ConversionManager::check_size_and_type_recurse(
        py::handle node,
        deg_t depth
)
{
    RPY_CHECK(depth <= m_options.max_nested);
    if (py::hasattr(node, "__dlpack__")) {
        // DLPack arguments have their inner-most dimension as leaves
        check_dl_size(py_to_dlpack(node), depth);
    } else if (py::isinstance<py::buffer>(node)) {
        // buffer arguments have their inner-most dimension as leaves
        check_buffer_size(py::reinterpret_borrow<py::buffer>(node), depth);
    } else if (py::isinstance<py::dict>(node)) {
        auto as_dict = py::reinterpret_borrow<py::dict>(node);
        RPY_CHECK(as_dict.size() > 0);

        bool is_leaf
                = std::all_of(as_dict.begin(), as_dict.end(), [this](auto kv) {
                      if (!is_pyscalar(kv.second)) { return false; }

                      return !static_cast<bool>(
                              (!py::isinstance<py::int_>(kv.first)
                               || m_options.alternative_key == nullptr
                               || !py::isinstance(
                                       kv.first,
                                       m_options.alternative_key->py_key_type
                               ))
                      );
                  });

        if (is_leaf) {
            auto& leaf = add_leaf(as_dict, LeafType::Dict);
            leaf.value_type = ValueType::KeyValue;
            leaf.shape.push_back(py::len(node));
            leaf.scalar_info
                    = py_type_to_type_info(py::type::of(as_dict.begin()->second)
                    );
        } else if (m_options.allow_timestamped && depth == 0) {
            m_options.indices.reserve(py::len(as_dict));
            for (const auto ts_value : as_dict) {
                // check if this is timestamped data. This is allowed only at
                // depth 0 if enabled by the options
                RPY_CHECK(py::isinstance<py::float_>(ts_value.first));
                m_options.indices.push_back(ts_value.first.cast<param_t>());
                check_size_and_type_recurse(ts_value.second, depth + 1);
            }
        } else {
            RPY_THROW(
                    py::value_error,
                    "dict must be key-scalar or timestamp-value"
            );
        }
    } else if (py::isinstance<py::sequence>(node)) {
        RPY_CHECK(py::len(node) > 0);
        optional<ValueType> expected_tp;
        bool is_leaf = std::all_of(
                node.begin(),
                node.end(),
                [this, &expected_tp](auto item) {
                    if (py::isinstance<py::tuple>(item)) {
                        if (expected_tp) {
                            if (*expected_tp != ValueType::KeyValue) {
                                RPY_THROW(
                                        py::value_error,
                                        "mismatched scalar and key-scalar data"
                                );
                            }
                        } else {
                            expected_tp = ValueType::KeyValue;
                        }
                        if (py::len(item) != 2) { return false; }

                        if (!is_pyscalar(item[py::int_(1)])) { return false; }

                        auto key = item[py::int_(0)];
                        return !static_cast<bool>(
                                !py::isinstance<py::int_>(key)
                                || m_options.alternative_key == nullptr
                                || !py::isinstance(
                                        key,
                                        m_options.alternative_key->py_key_type
                                )
                        );
                    }

                    if (expected_tp) {
                        if (*expected_tp != ValueType::Value) {
                            RPY_THROW(
                                    py::value_error,
                                    "mismatched scalar and key-scalar data"
                            );
                        }
                    } else {
                        expected_tp = ValueType::Value;
                    }

                    return is_pyscalar(item);
                }
        );

        if (is_leaf) {
            RPY_DBG_ASSERT(expected_tp);
            auto& leaf = add_leaf(node, LeafType::Sequence);
            leaf.value_type = *expected_tp;
            leaf.shape.push_back(py::len(node));
            leaf.scalar_info = py_type_to_type_info(py::type::of(node));
        } else if (m_options.allow_timestamped && depth == 0) {
            for (auto pair : node) {
                RPY_CHECK(
                        py::isinstance<py::tuple_>(pair) && py::len(pair) == 2
                );
                auto ts = pair[py::int_(0)];
                auto val = pair[py::int_(1)];
                RPY_CHECK(py::isinstance<py::float_>(ts));
                m_options.indices.push_back(ts.cast<param_t>());
                check_size_and_type_recurse(val, depth + 1);
            }
        } else {
            RPY_THROW(
                    py::value_error,
                    "expected sequence of scalars, key-scalar pairs, array, "
                    "dict, or sequence"
            );
        }
    } else {
        RPY_THROW(
                py::type_error,
                "expected array (buffer/dlpack), dict, or sequence"
        );
    }
}
void ConversionManager::check_size_and_type(py::handle arg)
{
    deg_t depth = 0;
    optional<ValueType> expected_type;

    check_size_and_type_recurse(arg, depth);
}

/**
 * \class ConversionManager
 *
 * \brief ConversionManager class is responsible for parsing and converting
 *Python arguments to C++ types.
 *
 * The ConversionManager class provides various methods for handling different
 *types of Python objects, including scalars, keys, lie groups, DLTensor
 *objects, buffers, dicts, and sequences . It also performs size and type
 *checking and manages the conversion process.
 *
 * The ConversionManager class is part of the encapsulating struct
 *ConversionManager, which also includes the following member variables:
 * - m_data: A KeyScalarArray representing the data being converted.
 * - m_options: A reference to a DataArgOptions object.
 * - m_offset: An offset used for computing the key.
 * - m_leaves: A small_vector of LeafItem objects.
 *
 * \see ConversionManager::parse_argument(py::handle arg)
 *
 */
void ConversionManager::parse_argument(py::handle arg)
{
    check_size_and_type(arg);
}
