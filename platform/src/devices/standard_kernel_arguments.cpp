//
// Created by sam on 01/07/24.
//

#include "standard_kernel_arguments.h"

#include <roughpy/core/errors.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/ranges.h>

#include "host_device.h"
#include "buffer.h"

using namespace rpy;
using namespace rpy::devices;

static constexpr string_view name_of(const KernelArgumentType& tp) noexcept
{
    switch (tp) {
        case KernelArgumentType::ResultBuffer: return "result buffer";
        case KernelArgumentType::ArgBuffer: return "argument buffer";
        case KernelArgumentType::ResultValue: return "result value";
        case KernelArgumentType::ArgValue: return "argument value";
        case KernelArgumentType::Operator: return "operator";
    }
    RPY_UNREACHABLE_RETURN("");
}

StandardKernelArguments::~StandardKernelArguments()
{
    for (const auto& [tp, arg] : views::zip(m_signature, m_args)) {
        if (tp == KernelArgumentType::ResultBuffer
            || tp == KernelArgumentType::ArgBuffer) {
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

    if (m_signature[idx] == KernelArgumentType::ResultBuffer) {
        RPY_CHECK(buffer.mode() != BufferMode::Read);
    } else if (m_signature[idx] == KernelArgumentType::ArgBuffer) {
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

    if (RPY_LIKELY(idx == m_args.size())) {
        // Argument is the next one to be added
        add_type(buffer.type());
        m_args.push_back(KernelArg{static_cast<void*>(buffer.release())});
    } else if (idx == m_args.size()) {
        // Argument already exists, modify inplace.
        auto old_value = steal_cast(static_cast<BufferInterface*>(m_args[idx].ptr));
        // Buffers must have the same underlying type
        RPY_CHECK(buffer.type() == old_value.type());
        m_args[idx].ptr = buffer.release();
    } else {
        RPY_THROW(std::runtime_error, "Unable to modify argument");
    }

}
void StandardKernelArguments::update_arg(dimn_t idx, ConstReference value)
{
    RPY_CHECK(idx < m_signature.size());
    RPY_CHECK(
            m_signature[idx] == KernelArgumentType::ArgValue,
            string_cat("Expected ", name_of(m_signature[idx]), " but got value")
    );

    if (RPY_LIKELY(idx == m_args.size())) {
        // Argument is the next one to be added
        m_args.emplace_back(KernelArg{const_cast<void*>(value.data())});
        add_type(value.type());
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
            m_signature[idx] == KernelArgumentType::Operator,
            string_cat(
                    "Expected ",
                    name_of(m_signature[idx]),
                    " but got operator"
            )
    );

    if (RPY_LIKELY(idx == m_args.size())) {
        // Argument is the next one to be added
        m_args.emplace_back(KernelArg{const_cast<void*>(static_cast<const void*>(&op))});
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
