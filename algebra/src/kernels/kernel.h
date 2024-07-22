//
// Created by sam on 10/04/24.
//

#ifndef KERNEL_H
#define KERNEL_H

#include "common.h"

#include <roughpy/devices/device_handle.h>
#include <roughpy/devices/kernel.h>

#include "arg_data.h"
#include "key_algorithms.h"

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
    using base_t = typename FirstArgSpec::template data<
            BoundArgs<RemainingArgSpec...>>;

public:
    using base_t::eval_device;
    using base_t::eval_generic;
    using base_t::eval_host;
    using base_t::get_device;
    using base_t::get_suffix;
    using base_t::get_type;
    using base_t::get_type_id;
    using base_t::resize;
    using base_t::size;

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

    void resize(dimn_t size) { (void) this; }

    RPY_NO_DISCARD devices::Device get_device() const noexcept
    {
        return nullptr;
    }

    RPY_NO_DISCARD dimn_t size() const noexcept { return 0; }
    RPY_NO_DISCARD string get_suffix() const noexcept { return string(); }

    RPY_NO_DISCARD string_view get_type_id() const noexcept { return ""; }

    RPY_NO_DISCARD scalars::TypePtr get_type() const
    {
        RPY_THROW(std::runtime_error, "cannot determine scalar type");
    }
};

RPY_NO_DISCARD optional<devices::Kernel> get_kernel(
        string_view kernel_name,
        string_view type_id,
        string_view suffix,
        const devices::Device& device
);

}// namespace dtl

template <typename Derived, typename... ArgSpec>
class VectorKernelBase
{
    const Basis* p_basis;

    template <typename Spec>
    using arg_type = typename Spec::arg_type;

    using BoundArgs = dtl::BoundArgs<ArgSpec...>;

protected:
    Derived& instance() noexcept { return static_cast<Derived&>(*this); }
    const Derived& instance() const noexcept
    {
        return static_cast<const Derived&>(*this);
    }

    bool eval_device(
            const devices::Device& device,
            string_view suffix,
            BoundArgs& bound
    ) const
    {
        if (auto kernel = dtl::get_kernel(
                    instance().kernel_name(),
                    bound.get_type_id(),
                    suffix,
                    device
            )) {
            devices::KernelLaunchParams params(
                    devices::Size3{bound.size()},
                    devices::Dim3{1}
            );
            bound.eval_device([k = *kernel, params](auto&&... args) {
                return k(params, std::forward<decltype(args)>(args)...);
            });
            return true;
        }
        return false;
    }
    bool eval_host(string_view suffix, BoundArgs& bound) const
    {
        if (auto kernel = dtl::get_kernel(
                    instance().kernel_name(),
                    bound.get_type_id(),
                    suffix,
                    bound.get_device()
            )) {
            devices::KernelLaunchParams params(
                    devices::Size3{bound.size()},
                    devices::Dim3{1}
            );
            bound.eval_host([k = *kernel, params](auto&&... args) {
                return k(params, std::forward<decltype(args)>(args)...);
            });
            return true;
        }
        return false;
    }
    bool eval_generic(BoundArgs& bound) const
    {
        dtl::GenericKernel<ArgSpec...> kernel(instance().generic_op(), p_basis);
        bound.eval_generic(kernel);
        return true;
    }

public:
    explicit VectorKernelBase(const Basis* basis) : p_basis(basis) {}

    void operator()(arg_type<ArgSpec>... spec) const;
};

template <typename Derived, typename... ArgSpec>
void VectorKernelBase<Derived, ArgSpec...>::operator()(arg_type<ArgSpec>... spec
) const
{
    BoundArgs bound_args(spec...);
    const auto suffix = bound_args.get_suffix();
    const auto device = bound_args.get_device();
    bound_args.resize(bound_args.size());
    if (device->is_host() || !eval_device(device, suffix, bound_args)) {
        if (!eval_host(suffix, bound_args)) { eval_generic(bound_args); }
    }
}

}// namespace algebra
}// namespace rpy

#endif// KERNEL_H