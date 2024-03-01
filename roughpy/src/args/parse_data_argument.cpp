//
// Created by sam on 2/26/24.
//

#include "parse_data_argument.h"

#include "buffer_info.h"
#include "dlpack.h"
#include "dlpack_helpers.h"
#include "numpy.h"

#include "scalars/r_py_polynomial.h"
#include "scalars/scalar.h"
#include "scalars/scalars.h"
#include "scalars/scalar_type.h"
#include "scalars/pytype_conversion.h"

#include <roughpy/algebra/algebra_fwd.h>
#include <roughpy/algebra/lie.h>
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


class ConversionManager
{
    scalars::KeyScalarArray m_data;
    DataArgOptions& m_options;
    dimn_t m_offset = 0;

public:
    explicit ConversionManager(DataArgOptions& options)
        : m_data(),
          m_options(options)
    {}

    scalars::KeyScalarArray take() && { return m_data; }

private:
    RPY_NO_DISCARD bool is_scalar(py::handle obj) const noexcept;
    RPY_NO_DISCARD bool is_sparse() const noexcept { return m_data.has_keys(); }

    RPY_NO_DISCARD key_type compute_key() const noexcept;
    RPY_NO_DISCARD key_type convert_key(py::handle pykey) const;
    RPY_NO_DISCARD scalars::Scalar convert_scalar(py::handle scalar) const;

    LeafItem& add_leaf(py::handle node, LeafType type);

    void push(scalars::Scalar scalar);
    void push(key_type key, scalars::Scalar scalar);
    void push(const algebra::Lie& lie);

    void compute_size_and_allocate();

    void handle_scalar(py::handle value);
    void handle_key_scalar(py::handle value);
    void handle_lie(py::handle value);
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

scalars::KeyScalarArray python::parse_data_argument(
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

LeafItem& ConversionManager::add_leaf(py::handle node, LeafType type)
{
    m_options.leaves.push_back(
            {{}, py::reinterpret_borrow<py::object>(node), 0, 0, {}, type}
    );
    return m_options.leaves.back();
}

void ConversionManager::push(scalars::Scalar scalar)
{
    RPY_DBG_ASSERT(m_offset < m_data.size());
    m_data[m_offset] = scalar;
    ++m_offset;
}
void ConversionManager::push(key_type key, scalars::Scalar scalar)
{
    RPY_DBG_ASSERT(m_data.has_keys() && m_offset < m_data.size());
    m_data.keys()[m_offset] = key;
    push(std::move(scalar));
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

void ConversionManager::compute_size_and_allocate()
{
    RPY_CHECK(!m_options.leaves.empty());
    m_data = scalars::KeyScalarArray(m_options.scalar_type);

    bool make_sparse = false;
    idimn_t size = 0;
    for (const auto& leaf : m_options.leaves) {
        size += static_cast<idimn_t>(leaf.size);
        make_sparse |= (leaf.value_type == ValueType::KeyValue);
    }

    if (size > 0) {
        m_data.allocate_scalars(size);
        if (make_sparse) { m_data.allocate_keys(size); }
    }
    RPY_DBG_ASSERT(m_offset == 0);
}

void ConversionManager::handle_scalar(py::handle value)
{
    if (is_sparse()) {
        push(compute_key(), convert_scalar(value));
    } else {
        push(convert_scalar(value));
    }
}

void ConversionManager::handle_key_scalar(py::handle value)
{
    RPY_DBG_ASSERT(is_sparse());
    RPY_DBG_ASSERT(py::isinstance<py::tuple>(value) && py::len(value) == 2);
    push(convert_key(value[py::int_(0)]),
         convert_scalar(value[py::int_(1)]));
}

void ConversionManager::handle_lie(py::handle value)
{
    push(value.cast<const algebra::Lie&>());
}

void ConversionManager::handle_dltensor_leaf(LeafItem& leaf)
{
    auto* managed = unpack_dl_capsule(leaf.object);
    const auto& tensor = managed->dl_tensor;

    const auto size = leaf.size;

    if (tensor.strides == nullptr) {
        scalars::ScalarArray dst_array = m_data[{m_offset, m_offset + size}];
        m_options.scalar_type->convert_copy(
                dst_array,
                {leaf.scalar_info, tensor.data, size}
        );
        m_offset += size;
        return;
    }

    boost::container::small_vector<idimn_t, 2> index(tensor.ndim);
    const auto* src = reinterpret_cast<const byte*>(tensor.data);

    // TODO: optimisation for when the inner-most parts are contiguous
    auto advance_index = [&index, &tensor]() {
        for (auto i = tensor.ndim - 1; i >= 0; --i) {
            if (index[i] < tensor.shape[i] - 1) {
                ++index[i];
                break;
            } else {
                index[i] = 0;
            }
        }
    };

    auto get_src_offset
            = [&index, &tensor, itemsize = leaf.scalar_info.bytes]() {
                  dimn_t idx = 0;
                  for (auto i = 0; i < tensor.ndim; ++i) {
                      idx += index[i] * tensor.strides[i];
                  }
                  return idx * itemsize;
              };

    auto get_dst_offset = [&index, &tensor]() {
        dimn_t idx = 0;
        for (auto i = 0; i < tensor.ndim; ++i) {
            idx += index[i] * tensor.shape[i];
        }
        return idx;
    };

    while (index[0] < tensor.shape[0]) {
        m_data[m_offset + get_dst_offset()]
                = scalars::Scalar(leaf.scalar_info, src + get_src_offset());
        advance_index();
    }

    m_offset += size;
}

void ConversionManager::handle_buffer_leaf(rpy::python::LeafItem& leaf)
{
    BufferInfo info(leaf.object.ptr());

    // This is substantially similar, but not identical, to the dlpack methods

    // The difference with the dlpack thing is that the strides of buffers are
    // in bytes rather than elements

    if (info.is_contiguous()) {
        auto dst_array = m_data[{m_offset, m_offset + leaf.size}];
        m_options.scalar_type->convert_copy(
                dst_array,
                {leaf.scalar_info, info.data(), leaf.size}
        );
    } else {
        auto index = info.new_index();

        // TODO: optimisation for when the inner-most parts are contiguous
        auto advance_index = [&index, &info]() {
            for (auto i = info.ndim() - 1; i >= 0; --i) {
                if (index[i] < info.shape()[i] - 1) {
                    ++index[i];
                    break;
                } else {
                    index[i] = 0;
                }
            }
        };

        auto get_dst_offset = [&index, &info]() {
            dimn_t idx = 0;
            for (auto i = 0; i < info.ndim(); ++i) {
                idx += index[i] * info.shape()[i];
            }
            return idx;
        };

        while (index[0] < info.shape()[0]) {
            m_data[m_offset + get_dst_offset()]
                    = scalars::Scalar(leaf.scalar_info, info.ptr(index.data()));
            advance_index();
        }
    }
    m_offset += leaf.size;
}

void ConversionManager::handle_dict_leaf(rpy::python::LeafItem& leaf)
{
    leaf.offset = m_offset;
    for (auto [key_o, scalar_o] : python::steal_as<py::dict>(leaf.object)) {
        push(convert_key(key_o), py_to_scalar(m_options.scalar_type, scalar_o));
    }
}

void ConversionManager::handle_sequence_leaf(rpy::python::LeafItem& leaf)
{
    leaf.offset = m_offset;
    if (leaf.value_type == ValueType::Value) {
        for (auto scalar : leaf.object) { handle_scalar(scalar); }
    } else {
        for (auto key_scalar : leaf.object) { handle_key_scalar(key_scalar); }
    }
}

void ConversionManager::do_conversion()
{
    compute_size_and_allocate();
    for (auto& leaf : m_options.leaves) {
        switch (leaf.leaf_type) {
            case LeafType::Scalar: handle_scalar(leaf.object); break;
            case LeafType::KeyScalar: handle_key_scalar(leaf.object); break;
            case LeafType::Lie: handle_lie(leaf.object); break;
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
            1,
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
            1,
            std::multiplies<>()
    ));
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
                      if (!is_scalar(kv.second)) { return false; }

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

                    return is_scalar(item);
                }
        );

        if (is_leaf) {
            RPY_DBG_ASSERT(expected_tp);
            auto& leaf = add_leaf(node, LeafType::Sequence);
            leaf.value_type = *expected_tp;
            leaf.shape.push_back(py::len(node));
//            leaf.scalar_info = py_type_to_type_info(py::type::of(node));
            leaf.scalar_info = devices::type_info<double>();
            leaf.size = leaf.shape[0];
        } else {
            for (auto pair : node) {
                if (m_options.allow_timestamped && depth == 0 &&
                    py::isinstance<py::tuple>(pair) && py::len(pair) == 2) {
                    auto ts = pair[py::int_(0)];
                    auto val = pair[py::int_(1)];
                    RPY_CHECK(py::isinstance<py::float_>(ts));
                    m_options.indices.push_back(ts.cast<param_t>());
                    check_size_and_type_recurse(val, depth + 1);
                } else {
                    check_size_and_type_recurse(pair, depth+1);
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
    deg_t depth = 0;
    optional<ValueType> expected_type;

    check_size_and_type_recurse(arg, depth);
}

const scalars::ScalarType* ConversionManager::compute_scalar_type() const
{
    if (m_options.scalar_type != nullptr) { return m_options.scalar_type; }

    auto info = devices::type_info<int>();

    for (const auto& leaf : m_options.leaves) {
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
    compute_size_and_allocate();
    if (!m_data.empty()) {
        do_conversion();
    }
}
