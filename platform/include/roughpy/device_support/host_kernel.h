//
// Created by sam on 30/04/24.
//

#ifndef ROUGHPY_DEVICES_HOST_KERNEL_H
#define ROUGHPY_DEVICES_HOST_KERNEL_H

#include <roughpy/core/traits.h>
#include <roughpy/devices/host_device.h>
#include <roughpy/devices/kernel.h>

#include <boost/compressed_pair.hpp>

namespace rpy {
namespace devices {

/**
 * @brief The HostKernel class represents a host kernel.
 *
 * The HostKernel class is a concrete class that implements the KernelInterface
 * interface. It allows for executing a host kernel.
 */
template <typename F, typename... ArgSpec>
class HostKernel : public dtl::RefCountBase<KernelInterface>
{
    using signature_t = StandardKernelSignature<ArgSpec...>;
    boost::compressed_pair<F, string> m_func_and_name;

    using ArgBinder = ArgumentBinder<typename signature_t::Parameter, F>;

    static_assert(
            std::tuple_size_v<args_t<F>> == signature_t::num_params,
            "Wrong number of arguments"
    );

public:
    HostKernel(F&& func, string name)
        : RefCountBase(StandardKernelSignature<ArgSpec...>::make()),
          m_func_and_name(std::move(func), std::move(name))
    {}

    explicit HostKernel(string name) : m_func_and_name(F(), std::move(name)) {}

    RPY_NO_DISCARD bool is_host() const noexcept override;
    RPY_NO_DISCARD DeviceType device_type() const noexcept override;
    RPY_NO_DISCARD Device device() const noexcept override;
    RPY_NO_DISCARD string name() const override;
    RPY_NO_DISCARD dimn_t num_args() const override;
    RPY_NO_DISCARD Event launch_kernel_async(
            Queue& queue,
            const KernelLaunchParams& params,
            const KernelArguments& args
    ) const override;
    EventStatus launch_kernel_sync(
            Queue& queue,
            const KernelLaunchParams& params,
            const KernelArguments& args
    ) const override;
};

template <typename F, typename... ArgSpec>
bool HostKernel<F, ArgSpec...>::is_host() const noexcept
{
    return true;
}
template <typename F, typename... ArgSpec>
DeviceType HostKernel<F, ArgSpec...>::device_type() const noexcept
{
    return DeviceType::CPU;
}
template <typename F, typename... ArgSpec>
Device HostKernel<F, ArgSpec...>::device() const noexcept
{
    return get_host_device();
}
template <typename F, typename... ArgSpec>
string HostKernel<F, ArgSpec...>::name() const
{
    return m_func_and_name.second;
    ;
}
template <typename F, typename... ArgSpec>
dimn_t HostKernel<F, ArgSpec...>::num_args() const
{
    return signature().num_parameters();
}

template <typename F, typename... ArgSpec>
Event HostKernel<F, ArgSpec...>::launch_kernel_async(
        Queue& queue,
        const KernelLaunchParams& params,
        const KernelArguments& args
) const
{
    ArgBinder::eval(m_func_and_name.first(), params, args);
    return Event();
}
template <typename F, typename... ArgSpec>
EventStatus HostKernel<F, ArgSpec...>::launch_kernel_sync(
        Queue& queue,
        const KernelLaunchParams& params,
        const KernelArguments& args
) const
{
    ArgBinder::eval(m_func_and_name.first(), params, args);
    return EventStatus::CompletedSuccessfully;
}

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICES_HOST_KERNEL_H
