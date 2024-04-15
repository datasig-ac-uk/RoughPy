//
// Created by sam on 10/04/24.
//

#ifndef KERNEL_H
#define KERNEL_H


#include "common.h"

#include <roughpy/scalars/devices/device_handle.h>
#include <roughpy/scalars/devices/kernel.h>

#include "key_algorithms.h"
#include "mutable_vector_element.h"
#include "arg_data.h"


namespace rpy {
namespace algebra {

namespace dtl {

using KArg = devices::KernelArgument;



class KernelLaunchData
{
    devices::KernelLaunchParams m_params;
    string m_suffix;
    string_view m_name;
};

template <typename... ArgSpec>
class BoundArgs;

template <typename FirstArgSpec, typename... RemainingArgSpec>
class BoundArgs<FirstArgSpec, RemainingArgSpec...>
    : FirstArgSpec::template data<BoundArgs<RemainingArgSpec...>>
{
    using base_t = typename FirstArgSpec::
            template data<FirstArgSpec, BoundArgs<RemainingArgSpec...>>;

public:
    using base_t::eval_device;
    using base_t::eval_generic;
    using base_t::eval_host;
    using base_t::get_suffix;

    template <typename... Args>
    explicit BoundArgs(Args&&... args) : base_t(std::forward<Args>(args)...)
    {}
};

template <>
class BoundArgs<>
{

public:
    template <typename F>
    decltype(auto) eval_device(F&& func) const
    {
        return func();
    }
    template <typename F>
    decltype(auto) eval_host(F&& func) const
    {
        return func();
    }
    template <typename F>
    decltype(auto) eval_generic(F&& func) const
    {
        return func();
    }

    RPY_NO_DISCARD string get_suffix() const noexcept { return string(); }
};

RPY_NO_DISCARD optional<devices::Kernel> get_kernel(
        string_view kernel_name,
        string_view suffix,
        const devices::Device& device
);

}// namespace dtl




template <typename Derived, typename... ArgSpec>
class VectorKernelBase
{

    template <typename Spec>
    using arg_type = typename Spec::arg_type;

    using BoundArgs = dtl::BoundArgs<ArgSpec...>;

protected:
    Derived& instance() noexcept { return static_cast<Derived&>(*this); }
    const Derived& instance() const noexcept
    {
        return static_cast<const Derived&>(*this);
    }

    bool eval_device(string_view suffix, BoundArgs& bound)
    {
        if (auto kernel = dtl::get_kernel(
                    instance().kernel_name(),
                    suffix,
                    bound.get_device()
            )) {
            bound.eval_host(*kernel);
            return true;
        }
        return false;
    }
    bool eval_host(string_view suffix, BoundArgs& bound)
    {
        if (auto kernel = dtl::get_kernel(
                    instance().kernel_name(),
                    suffix,
                    bound.get_device()
            )) {
            bound.eval_host(*kernel);
            return true;
        }
        return false;
    }
    bool eval_generic(BoundArgs& bound)
    {
        bound.eval_generic(instance().get_generic());
        return true;
    }

public:
    void operator()(arg_type<ArgSpec>... spec) const
    {
        BoundArgs bound_args(spec);
        const auto suffix = bound_args.get_suffix();

        if (!eval_device(suffix, bound_args)) {
            if (!eval_host(suffix, bound_args)) { eval_generic(bound_args); }
        }
    }
};

}// namespace algebra
}// namespace rpy

#endif// KERNEL_H
