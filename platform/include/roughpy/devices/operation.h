//
// Created by sam on 7/11/24.
//

#ifndef ROUGHPY_DEVICES_OPERATION_H
#define ROUGHPY_DEVICES_OPERATION_H

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include "buffer.h"
#include "core.h"
#include "kernel.h"
#include "kernel_operators.h"
#include "kernel_parameters.h"
#include "queue.h"

namespace rpy {
namespace devices {

class ROUGHPY_DEVICES_EXPORT Operation
{
    string m_kernel_name;
    const KernelSignature* p_signature;

protected:
    explicit Operation(string name, const KernelSignature* signature)
        : m_kernel_name(std::move(name)),
          p_signature(signature)
    {}

    virtual ~Operation() = default;

public:
    RPY_NO_DISCARD string_view kernel_name() const noexcept
    {
        return m_kernel_name;
    }

    RPY_NO_DISCARD const KernelSignature& signature() const
    {
        return *p_signature;
    };

protected:
    RPY_NO_DISCARD virtual optional<Kernel>
    get_kernel(const KernelArguments& args) const;

    virtual EventStatus eval_generic(
            Queue& queue,
            const KernelLaunchParams& params,
            const KernelArguments& args
    ) const = 0;

    virtual optional<Event>
    eval(Queue& queue,
         const KernelLaunchParams& params,
         const KernelArguments& args) const;

public:
    template <typename... Args>
    EventStatus
    operator()(const KernelLaunchParams& params, Args&&... args) const;

    template <typename... Args>
    Event
    operator()(Queue& queue, const KernelLaunchParams& params, Args&&... args)
            const;
};

template <typename... Args>
EventStatus
Operation::operator()(const KernelLaunchParams& params, Args&&... args) const
{
    auto binding = signature().new_binding();
    dtl::bind_args(*binding, std::forward<Args>(args)...);
    Queue queue;
    auto event = this->eval(queue, params, *binding);

    EventStatus status;
    if (event) {
        event->wait();
        status = event->status();
    } else {
        status = this->eval_generic(queue, params, *binding);
    }
    return status;
}

template <typename... Args>
Event Operation::operator()(
        Queue& queue,
        const KernelLaunchParams& params,
        Args&&... args
) const
{
    auto binding = signature().new_binding();
    dtl::bind_args(*binding, std::forward<Args>(args)...);
    auto event = this->eval(queue, params, *binding);
    if (event) { return *event; }
    return Event::completed_event(this->eval_generic(queue, params, *binding));
}

template <typename... ArgSpec>
class RPY_LOCAL StandardOperation : public Operation
{

protected:
    using signature_t = StandardKernelSignature<ArgSpec...>;
    explicit StandardOperation(string name);
};

template <typename... Args>
StandardOperation<Args...>::StandardOperation(string name)
    : Operation(std::move(name), signature_t::make())
{}

#define RPY_MAKE_STANDARD_OPERATION_DECLARE(ClsName, EXPORT_NAME, ...)         \
    class RPY_LOCAL ClsName                                                    \
        : public ::rpy::devices::StandardOperation<__VA_ARGS__>                \
    {                                                                          \
        using base_t = ::rpy::devices::StandardOperation<__VA_ARGS__>;         \
                                                                               \
    protected:                                                                 \
        using base_t::base_t;                                                  \
        using typename base_t::signature_t;                                    \
                                                                               \
        ::rpy::devices::EventStatus eval_generic(                              \
                ::rpy::devices::Queue& queue,                                  \
                const ::rpy::devices::KernelLaunchParams& params,              \
                const ::rpy::devices::KernelArguments& args                    \
        ) const override;                                                      \
                                                                               \
        RPY_NO_DISCARD EXPORT_NAME static const ::rpy::devices::Operation&     \
        get() noexcept;                                                        \
    };

#define RPY_MAKE_STANDARD_OPERATION_IMPL(ClsName, Name)                        \
    const ::rpy::devices::Operation& ClsName::get() noexcept                   \
    {                                                                          \
        static const ClsName instance(Name);                                   \
        return instance;                                                       \
    }                                                                          \
                                                                               \
    ::rpy::devices::EventStatus ClsName::eval_generic(                         \
            ::rpy::devices::Queue& queue,                                      \
            const ::rpy::devices::KernelLaunchParams& params,                  \
            const ::rpy::devices::KernelArguments& args                        \
    ) const

}// namespace devices
}// namespace rpy
#endif// ROUGHPY_DEVICES_OPERATION_H
