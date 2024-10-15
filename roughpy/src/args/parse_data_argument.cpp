//
// Created by sam on 2/26/24.
//

#include "parse_data_argument.h"

#include "buffer_info.h"
#include "dlpack.h"
#include "dlpack_helpers.h"
#include "numpy.h"
#include "strided_copy.h"

#include "scalars/pytype_conversion.h"
#include "scalars/r_py_polynomial.h"
#include "scalars/scalar.h"
#include "scalars/scalar_type.h"
#include "scalars/scalars.h"

#include <roughpy/algebra/algebra_fwd.h>
#include <roughpy/algebra/lie.h>
#include <roughpy/platform/devices/buffer.h>
#include <roughpy/platform/devices/device_handle.h>
#include <roughpy/scalars/key_scalar_array.h>
#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_type.h>

#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <boost/container/small_vector.hpp>

#include <algorithm>
#include <functional>
#include <numeric>
#include <utility>

using namespace rpy;
using namespace rpy::python;

namespace {

struct LeafItem {
    boost::container::small_vector<idimn_t, 1> shape;
    py::object object;
    dimn_t size;
    dimn_t offset;
    devices::TypeInfo scalar_info;
    LeafType leaf_type;
    ValueType value_type;
};

class ConversionManager
{
    std::vector<LeafItem> m_leaves;
    ParsedData m_parsed_data;
    DataArgOptions& m_options;
    dimn_t m_offset = 0;

public:
    explicit ConversionManager(DataArgOptions& options) : m_options(options) {}

    ParsedData take() && { return m_parsed_data; }

private:
    RPY_NO_DISCARD bool is_scalar(py::handle obj) const noexcept;

    RPY_NO_DISCARD key_type compute_key() const noexcept;
    RPY_NO_DISCARD key_type convert_key(py::handle pykey) const;
    RPY_NO_DISCARD scalars::Scalar convert_scalar(py::handle scalar) const;

    RPY_NO_DISCARD devices::TypeInfo type_info_from(py::handle scalar) const;

    LeafItem& add_leaf(py::handle node, LeafType type);

    void push(scalars::Scalar scalar);
    void push(key_type key, scalars::Scalar scalar);
    void push(const algebra::Lie& lie);

    LeafData& add_leaf(LeafItem& item);
    LeafData& allocate_leaf(LeafData& leaf);

    void handle_scalar(py::handle value, bool key = false);
    void handle_key_scalar(py::handle value);
    void handle_lie(py::handle value);

    void handle_scalar_leaf(LeafItem& leaf);
    void handle_key_scalar_leaf(LeafItem& leaf);
    void handle_lie_leaf(LeafItem& leaf);
    void handle_dltensor_leaf(LeafItem& leaf);
    void handle_buffer_leaf(LeafItem& leaf);
    void handle_dict_leaf(LeafItem& leaf);
    void handle_sequence_leaf(LeafItem& leaf);
    void do_conversion();

    void check_dl_size(py::capsule dlcap, deg_t depth);
    void check_buffer_size(py::buffer buffer, deg_t depth);

    void check_size_and_type_recurse(py::handle node, deg_t depth);
    void check_size_and_type(py::handle arg);

    RPY_NO_DISCARD const scalars::ScalarType* compute_scalar_type() const;

public:
    void parse_argument(py::handle arg);
};

}// namespace

ParsedData python::parse_data_argument(
        py::handle arg,
        rpy::python::DataArgOptions& options
)
{
    ConversionManager manager(options);
    manager.parse_argument(arg);
    return std::move(manager).take();
}

bool ConversionManager::is_scalar(py::handle obj) const noexcept
{
    if (py::isinstance<scalars::Scalar>(obj)) { return true; }

    if (py::isinstance<py::int_>(obj) || py::isinstance<py::float_>(obj)) {
        return true;
    }

    if (RPyPolynomial_Check(obj.ptr())) { return true; }

    if (m_options.check_imported_scalars) {
        if (is_imported_type(obj, "fractions", "Fraction")) { return true; }

        // TODO: Add more checks
    }

    return false;
}

key_type ConversionManager::compute_key() const noexcept { return m_offset; }

key_type ConversionManager::convert_key(py::handle pykey) const
{
    if (py::isinstance<py::int_>(pykey)) { return pykey.cast<key_type>(); }

    if (m_options.alternative_key != nullptr
        && py::isinstance(pykey, m_options.alternative_key->py_key_type)) {
        return m_options.alternative_key->converter(pykey);
    }

    RPY_THROW(py::type_error, "unrecognised key type");
}

scalars::Scalar ConversionManager::convert_scalar(py::handle scalar) const
{
    return py_to_scalar(m_options.scalar_type, scalar);
}

devices::TypeInfo ConversionManager::type_info_from(py::handle scalar) const
{
    if (py::isinstance<scalars::Scalar>(scalar)) {
        return scalar.cast<const scalars::Scalar&>().type_info();
    }

    if (RPyPolynomial_Check(scalar.ptr())) {
        return (*scalars::get_type("RationalPoly"))->type_info();
    }

    return py_type_to_type_info(py::type::of(scalar));
}

LeafItem& ConversionManager::add_leaf(py::handle node, LeafType type)
{
    m_leaves.push_back(
            {{}, py::reinterpret_borrow<py::object>(node), 0, 0, {}, type}
    );
    return m_leaves.back();
}

void ConversionManager::push(scalars::Scalar scalar)
{
    auto& data = m_parsed_data.back().data;
    RPY_DBG_ASSERT(m_offset < data.size());
    data[m_offset] = scalar;
    ++m_offset;
}
void ConversionManager::push(key_type key, scalars::Scalar scalar)
{
    auto& data = m_parsed_data.back().data;
    data.keys()[m_offset] = key;
    push(std::move(scalar));
}
void ConversionManager::push(const algebra::Lie& lie_data)
{
    if ((lie_data.storage_type() == algebra::VectorType::Dense)) {
        const auto& dense_data = lie_data.dense_data();

        m_offset += dense_data->size();
    } else {
        for (auto item : lie_data) { push(item.key(), item.value()); }
    }
}

LeafData& ConversionManager::add_leaf(LeafItem& item)
{
    auto& new_leaf = m_parsed_data.emplace_back(
            std::move(item.shape),
            scalars::KeyScalarArray(m_options.scalar_type),
            py::reinterpret_borrow<py::object>(item.object),
            item.size,
            item.scalar_info,
            item.leaf_type,
            item.value_type
    );

    m_offset = 0;
    return new_leaf;
}

LeafData& ConversionManager::allocate_leaf(LeafData& leaf)
{
    leaf.data = scalars::ScalarArray(m_options.scalar_type, leaf.size);
    if (leaf.value_type == ValueType::KeyValue) {
        leaf.data.allocate_keys(leaf.size);
    }
    return leaf;
}

void ConversionManager::handle_scalar(py::handle value, bool key)
{
    if (key) {
        push(compute_key(), convert_scalar(value));
    } else {
        push(convert_scalar(value));
    }
}

void ConversionManager::handle_key_scalar(py::handle value)
{
    RPY_DBG_ASSERT(py::isinstance<py::tuple>(value) && py::len(value) == 2);
    push(convert_key(value[py::int_(0)]), convert_scalar(value[py::int_(1)]));
}

void ConversionManager::handle_lie(py::handle value)
{
    push(value.cast<const algebra::Lie&>());
}

void ConversionManager::handle_scalar_leaf(LeafItem& leaf)
{
    auto& new_leaf = add_leaf(leaf);
    allocate_leaf(new_leaf);
    handle_scalar(new_leaf.owning_object);
}

void ConversionManager::handle_key_scalar_leaf(LeafItem& leaf)
{
    auto& new_leaf = add_leaf(leaf);
    allocate_leaf(new_leaf);
    handle_key_scalar(new_leaf.owning_object);
}
void ConversionManager::handle_lie_leaf(LeafItem& leaf)
{
    auto& new_leaf = add_leaf(leaf);
    const auto& leaf_lie = new_leaf.owning_object.cast<const algebra::Lie&>();
    if (leaf_lie.storage_type() == algebra::VectorType::Dense) {
        if (leaf_lie.coeff_type() == m_options.scalar_type) {
            // Borrowing is fine here.
            new_leaf.data = *leaf_lie.dense_data();
        } else {
            allocate_leaf(new_leaf);
            m_options.scalar_type->convert_copy(
                    new_leaf.data,
                    *leaf_lie.dense_data()
            );
        }
    } else {
        allocate_leaf(new_leaf);
        for (auto&& item : leaf_lie) { push(item.key(), item.value()); }
    }
}

void ConversionManager::handle_dltensor_leaf(LeafItem& leaf)
{
    auto& new_leaf = add_leaf(leaf);

    auto* managed = unpack_dl_capsule(new_leaf.owning_object);
    const auto& tensor = managed->dl_tensor;
    const auto size = new_leaf.size;

    auto borrow = is_C_contiguous<int64_t>(
            {tensor.strides, static_cast<dimn_t>(tensor.ndim)},
            {tensor.shape, static_cast<dimn_t>(tensor.ndim)},
            leaf.scalar_info.bytes
    );
    borrow &= m_options.scalar_type->type_info() == leaf.scalar_info;

    if (borrow) {
        new_leaf.data = scalars::ScalarArray(
                m_options.scalar_type,
                tensor.data,
                size
        );
        return;
    }

    allocate_leaf(new_leaf);

    boost::container::small_vector<idimn_t, 2> modified_shape(
            tensor.shape,
            tensor.shape + tensor.ndim
    );

    boost::container::small_vector<idimn_t, 2> modified_strides;
    if (tensor.strides != nullptr) {
        modified_strides.reserve(tensor.ndim);
        for (int32_t i = 0; i < tensor.ndim; ++i) {
            modified_strides.emplace_back(
                    tensor.strides[i] * leaf.scalar_info.bytes
            );
        }
    }

    python::stride_copy(
            new_leaf.data,
            {leaf.scalar_info, tensor.data, size},
            tensor.ndim,
            modified_shape.data(),
            modified_strides.empty() ? nullptr : modified_strides.data()
    );
}

void ConversionManager::handle_buffer_leaf(LeafItem& leaf)
{
    auto& new_leaf = add_leaf(leaf);

    BufferInfo info(new_leaf.owning_object.ptr());

    bool borrow = info.is_contiguous()
            && m_options.scalar_type->type_info() == new_leaf.origin_type_info;
    if (borrow) {
        new_leaf.data = scalars::ScalarArray(
                m_options.scalar_type,
                info.data(),
                info.size()
        );
        return;
    }

    allocate_leaf(new_leaf);

    python::stride_copy(
            new_leaf.data,
            {leaf.scalar_info, info.data(), static_cast<dimn_t>(info.size())},
            info.ndim(),
            info.shape(),
            info.strides()
    );
}

void ConversionManager::handle_dict_leaf(LeafItem& leaf)
{
    auto& new_leaf = add_leaf(leaf);
    allocate_leaf(new_leaf);

    for (auto [key_o, scalar_o] :
         py::reinterpret_borrow<py::dict>(new_leaf.owning_object)) {
        push(convert_key(key_o), py_to_scalar(m_options.scalar_type, scalar_o));
    }
}

void ConversionManager::handle_sequence_leaf(LeafItem& leaf)
{
    auto& new_leaf = add_leaf(leaf);
    allocate_leaf(new_leaf);

    if (leaf.value_type == ValueType::Value) {
        for (auto scalar : new_leaf.owning_object) { handle_scalar(scalar); }

    } else {
        for (auto key_scalar : new_leaf.owning_object) {
            handle_key_scalar(key_scalar);
        }
    }
}

void ConversionManager::do_conversion()
{
    for (auto& leaf : m_leaves) {
        switch (leaf.leaf_type) {
            case LeafType::Scalar: handle_scalar_leaf(leaf); break;
            case LeafType::KeyScalar: handle_key_scalar_leaf(leaf); break;
            case LeafType::Lie: handle_lie_leaf(leaf); break;
            case LeafType::DLTensor: handle_dltensor_leaf(leaf); break;
            case LeafType::Buffer: handle_buffer_leaf(leaf); break;
            case LeafType::Dict: handle_dict_leaf(leaf); break;
            case LeafType::Sequence: handle_sequence_leaf(leaf); break;
        }
    }
}

void ConversionManager::check_dl_size(py::capsule dlcap, deg_t depth)
{
    auto* managed_tensor = unpack_dl_capsule(dlcap);
    auto& tensor = managed_tensor->dl_tensor;

    depth += tensor.ndim;
    RPY_CHECK(depth <= m_options.max_nested);

    auto& leaf = add_leaf(dlcap, LeafType::DLTensor);

    leaf.shape.assign(tensor.shape, tensor.shape + tensor.ndim);

    leaf.value_type = ValueType::Value;
    leaf.scalar_info = convert_from_dl_datatype(tensor.dtype);

    leaf.size = static_cast<dimn_t>(std::accumulate(
            leaf.shape.begin(),
            leaf.shape.end(),
            1LL,
            std::multiplies<>()
    ));
}

void ConversionManager::check_buffer_size(py::buffer buffer, deg_t depth)
{
    const auto info = buffer.request();

    depth += info.ndim;
    RPY_CHECK(depth <= m_options.max_nested);

    auto& leaf = add_leaf(buffer, LeafType::Buffer);

    leaf.value_type = ValueType::Value;
    leaf.shape.reserve(info.ndim);
    for (const auto& dim : info.shape) { leaf.shape.push_back(dim); }

    leaf.scalar_info = py_buffer_to_type_info(info);
    leaf.size = static_cast<dimn_t>(std::accumulate(
            leaf.shape.begin(),
            leaf.shape.end(),
            1LL,
            std::multiplies<>()
    ));
}

void ConversionManager::check_size_and_type_recurse(
        py::handle node,
        deg_t depth
)
{
    RPY_CHECK(
            depth < m_options.max_nested,
            "maximum nested depth reached in this context",
            py::value_error
    );
    if (py::isinstance<algebra::Lie>(node)) {
        const auto& lie = node.cast<const algebra::Lie&>();
        auto& leaf = add_leaf(node, LeafType::Lie);
        leaf.shape.push_back(lie.dimension());
        leaf.value_type = lie.storage_type() == algebra::VectorType::Dense
                ? ValueType::Value
                : ValueType::KeyValue;
        leaf.size = lie.dimension();
        leaf.scalar_info = lie.coeff_type()->type_info();
    } else if (py::hasattr(node, "__dlpack__")) {
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
                      if (!is_scalar(kv.second)) { return false; }

                      return static_cast<bool>((
                              py::isinstance<py::int_>(kv.first)
                              || (m_options.alternative_key != nullptr
                                  && py::isinstance(
                                          kv.first,
                                          m_options.alternative_key->py_key_type
                                  ))
                      ));
                  });

        if (is_leaf) {
            auto& leaf = add_leaf(as_dict, LeafType::Dict);
            leaf.value_type = ValueType::KeyValue;
            leaf.shape.push_back(py::len(node));
            leaf.scalar_info = type_info_from(as_dict.begin()->second);
            leaf.size = leaf.shape[0];
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

                        if (!is_scalar(item[py::int_(1)])) { return false; }

                        auto key = item[py::int_(0)];
                        return static_cast<bool>(
                                py::isinstance<py::int_>(key)
                                || (m_options.alternative_key != nullptr
                                    && py::isinstance(
                                            key,
                                            m_options.alternative_key
                                                    ->py_key_type
                                    ))
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

                    return is_scalar(item);
                }
        );

        if (is_leaf) {
            RPY_DBG_ASSERT(expected_tp);
            auto& leaf = add_leaf(node, LeafType::Sequence);
            leaf.value_type = *expected_tp;
            leaf.shape.push_back(py::len(node));
            //            leaf.scalar_info =
            //            py_type_to_type_info(py::type::of(node));
            if (leaf.value_type == ValueType::Value) {
                leaf.scalar_info = type_info_from(node[py::int_(0)]);
            } else {
                leaf.scalar_info
                        = type_info_from(node[py::int_(0)][py::int_(1)]);
            }
            leaf.size = leaf.shape[0];
        } else {
            for (auto pair : node) {
                if (m_options.allow_timestamped && depth == 0
                    && py::isinstance<py::tuple>(pair) && py::len(pair) == 2) {
                    auto ts = pair[py::int_(0)];
                    auto val = pair[py::int_(1)];
                    RPY_CHECK(py::isinstance<py::float_>(ts));
                    m_options.indices.push_back(ts.cast<param_t>());
                    check_size_and_type_recurse(val, depth + 1);
                } else {
                    check_size_and_type_recurse(pair, depth + 1);
                }
            }
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
    if (m_options.allow_scalar && is_scalar(arg)) {
        auto& leaf = add_leaf(arg, LeafType::Scalar);
        leaf.size = 1;
        leaf.shape.push_back(1);
        leaf.value_type = ValueType::Value;
        leaf.scalar_info = type_info_from(arg);
        return;
    }

    if (m_options.allow_kv_pair && is_kv_pair(arg, m_options.alternative_key)) {
        auto& leaf = add_leaf(arg, LeafType::KeyScalar);
        leaf.size = 1;
        leaf.shape.push_back(1);
        leaf.value_type = ValueType::KeyValue;
        leaf.scalar_info = type_info_from(arg[py::int_(1)]);
        return;
    }

    deg_t depth = 0;
    check_size_and_type_recurse(arg, depth);
}

const scalars::ScalarType* ConversionManager::compute_scalar_type() const
{
    if (m_options.scalar_type != nullptr) { return m_options.scalar_type; }

    auto info = devices::type_info<int>();

    for (const auto& leaf : m_leaves) {
        info = scalars::compute_type_promotion(info, leaf.scalar_info);
    }

    return scalars::ScalarType::for_info(info);
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
    if (m_options.allow_none && arg.is_none()) { return; }
    check_size_and_type(arg);

    m_options.scalar_type = compute_scalar_type();
    do_conversion();
}

void ParsedData::fill_ks_stream(scalars::KeyScalarStream& ks_stream)
{
    for (const auto& leaf : *this) {
        switch (leaf.leaf_type) {
            case LeafType::Scalar:
                RPY_THROW(std::runtime_error, "scalar value disallowed");
                break;
            case LeafType::KeyScalar:
                RPY_THROW(std::runtime_error, "key-scalar value disallowed");
                break;
            case LeafType::DLTensor:
            case LeafType::Buffer: {
                if (leaf.size == 0) { break; }
                if (leaf.shape.size() == 1) {
                    auto sz = leaf.size;
                    ks_stream.push_back(leaf.data.borrow());

                } else {
                    dimn_t sz = leaf.shape.back();
                    dimn_t offset1 = 0;
                    for (dimn_t inner = 0; inner < leaf.size; inner += sz) {
                        ks_stream.push_back(leaf.data[{offset1, offset1 + sz}]);
                        offset1 += sz;
                    }
                }
            } break;
            case LeafType::Dict:
            case LeafType::Lie:
            case LeafType::Sequence:
                ks_stream.push_back(leaf.data.borrow());
                break;
        }
    }
}

void ParsedData::fill_ks_buffer(scalars::KeyScalarArray& ks_array)
{
    if (empty()) { return; }

    const auto sz = size();
    if (sz == 1) {
        ks_array = std::move(front().data);
    } else {
        RPY_THROW(
                std::runtime_error,
                "unable to flatten array into single buffer"
        );
    }
}
