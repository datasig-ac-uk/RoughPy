//
// Created by user on 22/06/23.
//

#include "r_py_tick_construction_helper.h"

#include <roughpy/streams/tick_stream.h>

#include "args/convert_timestamp.h"
#include "args/parse_schema.h"

using namespace rpy;
using namespace rpy::streams;
using namespace pybind11::literals;

python::RPyTickConstructionHelper::RPyTickConstructionHelper()
    : m_ticks(), p_schema(new StreamSchema), b_schema_only(false) {
}
python::RPyTickConstructionHelper::RPyTickConstructionHelper(bool schema_only)
    : m_ticks(), p_schema(new StreamSchema), b_schema_only(schema_only) {
}

python::RPyTickConstructionHelper::RPyTickConstructionHelper(std::shared_ptr<streams::StreamSchema> schema, bool schema_only)
    : m_ticks(), p_schema(std::move(schema)), b_schema_only(schema_only) {
    RPY_CHECK(!schema_only || !p_schema->is_final());
}

void python::RPyTickConstructionHelper::add_increment_to_schema(string label, const py::kwargs &RPY_UNUSED_VAR kwargs) {
    p_schema->insert_increment(std::move(label));
}
void python::RPyTickConstructionHelper::add_value_to_schema(string label, const py::kwargs &RPY_UNUSED_VAR kwargs) {
    p_schema->insert_value(std::move(label));
}
void python::RPyTickConstructionHelper::add_categorical_to_schema(string label, py::object variant, const py::kwargs &RPY_UNUSED_VAR kwargs) {
    auto &channel = p_schema->insert_categorical(std::move(label));
    if (!variant.is_none()) {
        channel.insert_variant(variant.cast<string>());
    }
}

void python::RPyTickConstructionHelper::fail_timestamp_none() {
    throw py::value_error("timestamp cannot be None when constructing stream");
}
void python::RPyTickConstructionHelper::fail_data_none() {
    throw py::value_error("data cannot be None when constructing stream");
}


void python::RPyTickConstructionHelper::add_increment(const py::str &label, py::object timestamp, py::object data, const py::kwargs &kwargs) {
    auto lbl = label.cast<string>();
    if (!p_schema->is_final()) {
        add_increment_to_schema(lbl, kwargs);
    }

    if (b_schema_only) {
        // Do Nothing
    } else if (timestamp.is_none()) {
        fail_timestamp_none();
    } else if (data.is_none()) {
        fail_data_none();
    } else {
        m_ticks.push_back({
            lbl,
            python::convert_timestamp(timestamp),
            std::move(data),
            ChannelType::Increment
        });
    }
}
void python::RPyTickConstructionHelper::add_value(const py::str &label, py::object timestamp, py::object data, const py::kwargs &kwargs) {
    auto lbl = label.cast<string>();
    if (!p_schema->is_final()) {
        add_value_to_schema(lbl, kwargs);
    }

    if (b_schema_only) {
        // Do Nothing
    } else if (timestamp.is_none()){
        fail_timestamp_none();
    } else if (data.is_none()) {
        fail_data_none();
    } else {
        m_ticks.push_back({
            lbl,
            python::convert_timestamp(timestamp),
            std::move(data),
            ChannelType::Value
        });
    }
}
void python::RPyTickConstructionHelper::add_categorical(const py::str &label, py::object timestamp, py::object variant, const py::kwargs &kwargs) {
    auto lbl = label.cast<string>();
    if (!p_schema->is_final()) {
        add_categorical_to_schema(lbl, variant, kwargs);
    }

    if (b_schema_only) {
        // Do Nothing
    } else if (timestamp.is_none()) {
        fail_timestamp_none();
    } else if (variant.is_none()) {
        throw py::value_error("variant cannot be None when constructing stream");
    } else {
        m_ticks.push_back({lbl,
                           python::convert_timestamp(timestamp),
                           std::move(variant),
                           ChannelType::Categorical});
    }

}


void python::init_tick_construction_helper(py::module_ &m) {
    using helper_t = python::RPyTickConstructionHelper;

    py::class_<helper_t> klass(m, "TickStreamConstructionHelper");

    klass.def(py::init<>());
    klass.def(py::init<bool>(), "schema_only"_a);
    klass.def(py::init<std::shared_ptr<StreamSchema>, bool>(), "schema"_a, "schema_only"_a = false);

    klass.def("add_increment", &helper_t::add_increment, "label"_a, "timestamp"_a = py::none(), "data"_a = py::none());
    klass.def("add_value", &helper_t::add_value, "label"_a, "timestamp"_a = py::none(), "data"_a = py::none());
    klass.def("add_categorical", &helper_t::add_categorical, "label"_a, "timestamp"_a = py::none(), "variant"_a = py::none());
}
