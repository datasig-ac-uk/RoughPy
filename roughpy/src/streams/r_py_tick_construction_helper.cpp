//
// Created by user on 22/06/23.
//

#include "r_py_tick_construction_helper.h"

#include <roughpy/streams/tick_stream.h>

#include "args/convert_timestamp.h"
#include "args/parse_schema.h"

#include "py_schema_context.h"

using namespace rpy;
using namespace rpy::streams;
using namespace pybind11::literals;

python::RPyTickConstructionHelper::RPyTickConstructionHelper()
    : m_ticks(), p_schema(new StreamSchema), b_schema_only(false),
      m_reference_time(py::none()),
      m_time_conversion_options{PyDateTimeResolution::Seconds}
{}
python::RPyTickConstructionHelper::RPyTickConstructionHelper(bool schema_only)
    : m_ticks(), p_schema(new StreamSchema), b_schema_only(schema_only),
      m_reference_time(py::none()),
      m_time_conversion_options{PyDateTimeResolution::Seconds}
{}

python::RPyTickConstructionHelper::RPyTickConstructionHelper(
        std::shared_ptr<streams::StreamSchema> schema, bool schema_only
)
    : m_ticks(), p_schema(std::move(schema)), b_schema_only(schema_only),
      m_reference_time(py::none()),
      m_time_conversion_options{PyDateTimeResolution::Seconds}
{
    if (!p_schema->is_final() && p_schema->context() == nullptr) {
        p_schema->init_context<PySchemaContext>(m_time_conversion_options);
    }
    RPY_CHECK(!schema_only || !p_schema->is_final());
}

void python::RPyTickConstructionHelper::add_increment_to_schema(
        string label, const py::kwargs& RPY_UNUSED_VAR kwargs
)
{
    p_schema->insert_increment(std::move(label));
}
void python::RPyTickConstructionHelper::add_value_to_schema(
        string label, const py::kwargs& RPY_UNUSED_VAR kwargs
)
{
    p_schema->insert_value(std::move(label));
}
void python::RPyTickConstructionHelper::add_categorical_to_schema(
        string label, py::object variant,
        const py::kwargs& RPY_UNUSED_VAR kwargs
)
{
    auto& channel = p_schema->insert_categorical(std::move(label));
    if (!variant.is_none()) { channel.insert_variant(variant.cast<string>()); }
}

void python::RPyTickConstructionHelper::fail_timestamp_none()
{
    RPY_THROW(py::value_error, "timestamp cannot be None when constructing stream");
}
void python::RPyTickConstructionHelper::fail_data_none()
{
    RPY_THROW(py::value_error, "data cannot be None when constructing stream");
}

void python::RPyTickConstructionHelper::add_tick(
        string label, py::object timestamp, py::object data,
        streams::ChannelType type, const py::kwargs& RPY_UNUSED_VAR kwargs
)
{

    if (b_schema_only) {
        // Do Nothing
    } else if (timestamp.is_none()) {
        fail_timestamp_none();
    } else if (data.is_none()) {
        fail_data_none();
    } else {
        if (m_reference_time.is_none()) { m_reference_time = timestamp; }
        m_ticks.push_back(
                {label,
                 python::convert_delta_from_datetimes(
                         timestamp, m_reference_time, m_time_conversion_options
                 ),
                 std::move(data), type}
        );
    }
}

void python::RPyTickConstructionHelper::add_increment(
        const py::str& label, py::object timestamp, py::object data,
        const py::kwargs& kwargs
)
{
    auto lbl = label.cast<string>();
    if (!p_schema->is_final()) { add_increment_to_schema(lbl, kwargs); }

    add_tick(
            std::move(lbl), std::move(timestamp), std::move(data),
            streams::ChannelType::Increment, kwargs
    );
}
void python::RPyTickConstructionHelper::add_value(
        const py::str& label, py::object timestamp, py::object data,
        const py::kwargs& kwargs
)
{
    auto lbl = label.cast<string>();
    if (!p_schema->is_final()) { add_value_to_schema(lbl, kwargs); }

    add_tick(
            std::move(lbl), std::move(timestamp), std::move(data),
            streams::ChannelType::Value, kwargs
    );
}
void python::RPyTickConstructionHelper::add_categorical(
        const py::str& label, py::object timestamp, py::object variant,
        const py::kwargs& kwargs
)
{
    auto lbl = label.cast<string>();
    if (!p_schema->is_final()) {
        add_categorical_to_schema(lbl, variant, kwargs);
    }

    add_tick(
            label.cast<string>(), std::move(timestamp), std::move(variant),
            streams::ChannelType::Categorical, kwargs
    );
}

void python::init_tick_construction_helper(py::module_& m)
{
    using helper_t = python::RPyTickConstructionHelper;

    py::class_<helper_t> klass(m, "TickStreamConstructionHelper");

    klass.def(py::init<>());
    klass.def(py::init<bool>(), "schema_only"_a);
    klass.def(
            py::init<std::shared_ptr<StreamSchema>, bool>(), "schema"_a,
            "schema_only"_a = false
    );

    klass.def(
            "add_increment", &helper_t::add_increment, "label"_a,
            "timestamp"_a = py::none(), "data"_a = py::none()
    );
    klass.def(
            "add_value", &helper_t::add_value, "label"_a,
            "timestamp"_a = py::none(), "data"_a = py::none()
    );
    klass.def(
            "add_categorical", &helper_t::add_categorical, "label"_a,
            "timestamp"_a = py::none(), "variant"_a = py::none()
    );
}
