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

    using ArgBinder = ArgumentBinder<typename signature_t::ParamsList, F>;

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

namespace dtl {

template <typename Impl, typename... ArgSpec>
class HostKernelBase : public RefCountBase<KernelInterface>
{

protected:
    using siganture_t = StandardKernelSignature<ArgSpec...>;

public:
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

template <typename Impl, typename... ArgSpec>
bool HostKernelBase<Impl, ArgSpec...>::is_host() const noexcept
{
    return true;
}
template <typename Impl, typename... ArgSpec>
DeviceType HostKernelBase<Impl, ArgSpec...>::device_type() const noexcept
{
    return DeviceType::CPU;
}
template <typename Impl, typename... ArgSpec>
Device HostKernelBase<Impl, ArgSpec...>::device() const noexcept
{
    return get_host_device();
}
template <typename Impl, typename... ArgSpec>
string HostKernelBase<Impl, ArgSpec...>::name() const
{
    return Impl::get_name();
}
template <typename Impl, typename... ArgSpec>
dimn_t HostKernelBase<Impl, ArgSpec...>::num_args() const
{
    return siganture_t::num_args;
}
template <typename Impl, typename... ArgSpec>
Event HostKernelBase<Impl, ArgSpec...>::launch_kernel_async(
        Queue& queue,
        const KernelLaunchParams& params,
        const KernelArguments& args
) const
{
    using Binding = devices::ArgumentBinder<
            typename siganture_t::ParamsList,
            decltype(Impl::run)>;
    Binding::eval(Impl::run, params, args);
    return Event::completed_event(EventStatus::CompletedSuccessfully);
}
template <typename Impl, typename... ArgSpec>
EventStatus HostKernelBase<Impl, ArgSpec...>::launch_kernel_sync(
        Queue& queue,
        const KernelLaunchParams& params,
        const KernelArguments& args
) const
{
    using Binding = devices::ArgumentBinder<
            typename siganture_t::ParamsList,
            decltype(Impl::run)>;
    Binding::eval(
            [&params](auto&&... kargs) {
                Impl::run(params, std::forward<decltype(kargs)>(kargs)...);
            },
            params,
            args
    );
    return EventStatus::CompletedSuccessfully;
}

}// namespace dtl

template <template <typename> class Operator, typename T>
class UnaryHostKernel : public dtl::HostKernelBase<
                                UnaryHostKernel<Operator, T>,
                                params::ResultBuffer<T>,
                                params::Buffer<T>,
                                params::Operator<T>>
{
public:
    static void
    run(const KernelLaunchParams& params,
        Slice<T> result,
        Slice<const T> arg,
        Operator<T>&& op)
    {
        const auto dim = params.total_work_size();
        RPY_DBG_ASSERT(dim <= result.size());
        RPY_DBG_ASSERT(dim <= arg.size());

        const auto& work_dims = params.work_groups();
        const auto& group_size = params.work_groups();

        for (dimn_t ix = 0; ix < work_dims.x; ++ix) {
            for (dimn_t iy = 0; iy < work_dims.y; ++iy) {
                for (dimn_t iz = 0; iz < work_dims.z; ++iz) {
                    const auto index = (ix*group_size.y + iy)*group_size.z + iz;
                    result[index] = op(index);
                }
            }
        }
    }
};

template <template <typename> class Operator, typename T>
class UnaryInplaceHostKernel : public dtl::HostKernelBase<UnaryHostKernel<Operator, T>,
    params::ResultBuffer<T>, params::Operator<T>>
{
public:

    static void run(const KernelLaunchParams& params, Slice<T> buffer)
    {

        const auto dim = params.total_work_size();
        RPY_DBG_ASSERT(dim <= buffer.size());

        const auto& work_dims = params.work_groups();
        const auto& group_size = params.work_groups();

        for (dimn_t ix = 0; ix < work_dims.x; ++ix) {
            for (dimn_t iy = 0; iy < work_dims.y; ++iy) {
                for (dimn_t iz = 0; iz < work_dims.z; ++iz) {
                    const auto index
                            = (ix * group_size.y + iy) * group_size.z + iz;
                    buffer[index] = op(buffer[index]);
                }
            }
        }
    }
};

template <template <typename> class Operator, typename T>
class BinaryHostKernel : public dtl::HostKernelBase<
                                 BinaryHostKernel<Operator, T>,
                                 params::ResultBuffer<T>,
                                 params::Buffer<T>,
                                 params::Buffer<T>,
                                 params::Operator<T>>
{
public:
    static void
    run(const KernelLaunchParams& params,
        Slice<T> result,
        Slice<const T> left,
        Slice<const T> right,
        Operator<T>&& op)
    {
        const auto dim = params.total_work_size();
        RPY_DBG_ASSERT(dim <= result.size());
        RPY_DBG_ASSERT(dim <= left.size());
        RPY_DBG_ASSERT(dim <= right.size());

        const auto& work_dims = params.work_groups();
        const auto& group_size = params.work_groups();

        // TODO: Add different branches for parameters where the number of dims
        // is 1/2?

        for (dimn_t ix = 0; ix < work_dims.x; ++ix) {
            for (dimn_t iy = 0; iy < work_dims.y; ++iy) {
                for (dimn_t iz = 0; iz < work_dims.z; ++iz) {
                    const auto index
                            = (ix * group_size.y + iy) * group_size.y + iz;
                    result[index] = op(left[index], right[index]);
                }
            }
        }
    }
};

template <template <typename> class Operator, typename T>
class BinaryInplaceHostKernel : public dtl::HostKernelBase<
                                        BinaryInplaceHostKernel<Operator, T>,
                                        params::ResultBuffer<T>,
                                        params::Buffer<T>,
                                        params::Operator<T>>
{
public:
    static void
    run(const KernelLaunchParams& params,
        Slice<T> result_left,
        Slice<const T> right,
        Operator<T>&& op)
    {
        const auto dim = params.total_work_size();
        RPY_DBG_ASSERT(dim <= result_left.size());
        RPY_DBG_ASSERT(dim <= right.size());

        const auto& work_dims = params.work_groups();
        const auto& group_size = params.work_groups();

        for (dimn_t ix = 0; ix < work_dims.x; ++ix) {
            for (dimn_t iy = 0; iy < work_dims.y; ++iy) {
                for (dimn_t iz = 0; iz < work_dims.z; ++iz) {
                    const auto index
                            = (ix * group_size.y + iy) * group_size.z + iz;
                    result_left[index] = op(result_left[index], right[index]);
                }
            }
        }
    }
};

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICES_HOST_KERNEL_H
