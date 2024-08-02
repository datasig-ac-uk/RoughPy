//
// Created by sam on 01/07/24.
//

#include "standard_kernel_arguments.h"

#include <roughpy/core/errors.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/ranges.h>

#include "buffer.h"
#include "host_device.h"
#include "kernel_parameters.h"

using namespace rpy;
using namespace rpy::devices;

StandardKernelArguments::StandardKernelArguments(
        const KernelSignature& signature
)
{
    m_args.reserve(signature.num_parameters());

    {
        const auto sig_types = signature.types();
        m_types.assign(sig_types.begin(), sig_types.end());
    }

    {
        const auto params = signature.parameters();
        m_signature.assign(params.begin(), params.end());
    }
}

static constexpr string_view
name_of(const typename KernelSignature::Parameter& tp) noexcept
{
    switch (tp.param_kind) {
        case static_cast<uint8_t>(params::ParameterType::ResultBuffer):
            return "result buffer";
        case static_cast<uint8_t>(params::ParameterType::ArgBuffer):
            return "argument buffer";
        case static_cast<uint8_t>(params::ParameterType::ResultValue):
            return "result value";
        case static_cast<uint8_t>(params::ParameterType::ArgValue):
            return "argument value";
        case static_cast<uint8_t>(params::ParameterType::Operator):
            return "operator";
        default: return "unknown";
    }
    RPY_UNREACHABLE_RETURN("");
}

void StandardKernelArguments::check_or_set_type(
        const dimn_t index,
        const TypePtr& type
)
{
    RPY_CHECK(index < m_types.size());
    auto& tp = m_types[index];
    if (!tp) {
        // Type is not yet set. Set the type
        tp = type;
        m_generics.push_back(&*tp);
    } else if (tp != type) {
        // Type is set and it does not match
        RPY_THROW(
                std::invalid_argument,
                string_cat(
                        "Unable to bind ",
                        type->name(),
                        " to parameter with type ",
                        tp->name()
                )
        );
    }
}

StandardKernelArguments::~StandardKernelArguments()
{
    for (const auto& [tp, arg] : views::zip(m_signature, m_args)) {
        if (tp.param_kind == params::ParameterType::ResultBuffer
            || tp.param_kind == params::ParameterType::ArgBuffer) {
            // make sure all the buffer arguments are properly cleaned up.
            auto retake_ownership
                    = steal_cast(static_cast<BufferInterface*>(arg.ptr));
            (void) retake_ownership;
        }
    }
}

containers::Vec<void*> StandardKernelArguments::raw_pointers() const
{
    containers::Vec<void*> result;
    result.reserve(m_args.size());

    for (const auto& arg : m_args) { result.push_back(arg.ptr); }

    return result;
}
Device StandardKernelArguments::get_device() const noexcept
{
    return get_host_device();
}
Slice<const TypePtr> StandardKernelArguments::get_types() const noexcept
{
    return {m_types.data(), m_types.size()};
}

Slice<const Type* const>
StandardKernelArguments::get_generic_types() const noexcept
{
    return {m_generics.data(), m_generics.size()};
}
containers::SmallVec<dimn_t, 2>
StandardKernelArguments::get_sizes() const noexcept
{
    containers::SmallVec<dimn_t, 2> result;
    result.reserve(m_args.size());
    for (const auto& [tp, arg] : views::zip(m_signature, m_args)) {
        if (tp.param_kind == params::ParameterType::ResultBuffer
            || tp.param_kind == params::ParameterType::ArgBuffer) {
            result.push_back(static_cast<const BufferInterface*>(arg.ptr)->size());
        }
    }
    return result;
}

void StandardKernelArguments::bind(Buffer buffer)
{
    update_arg(m_args.size(), std::move(buffer));
}
void StandardKernelArguments::bind(ConstReference value)
{
    update_arg(m_args.size(), value);
}
void StandardKernelArguments::bind(const operators::Operator& op)
{
    update_arg(m_args.size(), op);
}
void StandardKernelArguments::update_arg(dimn_t idx, Buffer buffer)
{
    RPY_CHECK(idx < m_signature.size());

    if (m_signature[idx].param_kind == params::ParameterType::ResultBuffer) {
        RPY_CHECK(buffer.mode() != BufferMode::Read);
    } else if (m_signature[idx].param_kind == params::ParameterType::ArgBuffer) {
        RPY_CHECK(buffer.mode() != BufferMode::Write);
    } else {
        RPY_THROW(
                std::runtime_error,
                string_cat(
                        "Expected ",
                        name_of(m_signature[idx]),
                        " but got buffer"
                )
        );
    }

    const auto& param_data = m_signature[idx];
    check_or_set_type(param_data.type_index, buffer.type());

    if (RPY_LIKELY(idx == m_args.size())) {
        // Argument is the next one to be added
        m_args.push_back(KernelArg{static_cast<void*>(buffer.release())});
    } else if (idx < m_args.size()) {
        // Argument already exists, modify inplace.
        auto old_value
                = steal_cast(static_cast<BufferInterface*>(m_args[idx].ptr));
        m_args[idx].ptr = buffer.release();
    } else {
        RPY_THROW(std::runtime_error, "Unable to modify argument");
    }
}
void StandardKernelArguments::update_arg(dimn_t idx, ConstReference value)
{
    RPY_CHECK(idx < m_signature.size());
    RPY_CHECK(
            m_signature[idx].param_kind == params::ParameterType::ArgValue,
            string_cat("Expected ", name_of(m_signature[idx]), " but got value")
    );

    const auto& param_data = m_signature[idx];
    check_or_set_type(param_data.type_index, value.type());

    if (RPY_LIKELY(idx == m_args.size())) {
        // Argument is the next one to be added
        m_args.emplace_back(KernelArg{const_cast<void*>(value.data())});
    } else if (idx < m_args.size()) {
        // Update an existing argument
        m_args[idx].ptr = const_cast<void*>(value.data());
    } else {
        RPY_THROW(std::runtime_error, "Unable to modify argument");
    }
}
void StandardKernelArguments::update_arg(
        dimn_t idx,
        const operators::Operator& op
)
{
    RPY_CHECK(idx < m_signature.size());
    RPY_CHECK(
            m_signature[idx].param_kind == params::ParameterType::Operator,
            string_cat(
                    "Expected ",
                    name_of(m_signature[idx]),
                    " but got operator"
            )
    );

    if (RPY_LIKELY(idx == m_args.size())) {
        // Argument is the next one to be added
        m_args.emplace_back(
                KernelArg{const_cast<void*>(static_cast<const void*>(&op))}
        );
    } else if (idx < m_args.size()) {
        // Update an existing argument
        m_args[idx].ptr = const_cast<void*>(static_cast<const void*>(&op));
    } else {
        RPY_THROW(std::runtime_error, "Unable to modify argument");
    }
}
dimn_t StandardKernelArguments::num_args() const noexcept
{
    return m_signature.size();
}
dimn_t StandardKernelArguments::true_num_args() const noexcept
{
    return m_signature.size();
}

dimn_t StandardKernelArguments::num_bound_args() const noexcept
{
    return m_args.size();
}

const Type* StandardKernelArguments::get_type(dimn_t index) const
{
    RPY_CHECK(index < m_args.size());
    return m_types[static_cast<dimn_t>(m_signature[index].type_index)].get();
}
void* StandardKernelArguments::get_raw_ptr(dimn_t index) const
{
    RPY_CHECK(index < m_args.size());
    return m_args[index].ptr;
}

Buffer StandardKernelArguments::get_buffer(dimn_t index) const
{
    RPY_CHECK(
            m_signature[index].param_kind == params::ParameterType::ArgBuffer
                    || m_signature[index].param_kind
                            == params::ParameterType::ResultBuffer,
            string_cat(
                    "requested buffer but param at index ",
                    std::to_string(index),
                    " is of type ",
                    name_of(m_signature[index])
            )
    );
    return Buffer(static_cast<BufferInterface*>(m_args[index].ptr));
}
ConstReference StandardKernelArguments::get_value(dimn_t index) const
{
    RPY_CHECK(
            m_signature[index].param_kind == params::ParameterType::ArgValue,
            string_cat(
                    "requested value but param at index ",
                    std::to_string(index),
                    " is of type ",
                    name_of(m_signature[index])
            )
    );
    return ConstReference(m_args[index].ptr, m_types[m_signature[index].type_index]);
}
const operators::Operator& StandardKernelArguments::get_operator(dimn_t index
) const
{
    RPY_CHECK(
            m_signature[index].param_kind == params::ParameterType::Operator,
            string_cat(
                    "requested operator but param at index ",
                    std::to_string(index),
                    " is of type ",
                    name_of(m_signature[index])
            )
    );
    return *static_cast<const operators::Operator*>(m_args[index].ptr);
}
