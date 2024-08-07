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

namespace dtl {

template <typename Impl, typename... ArgSpec>
class HostKernelBase : public RefCountBase<KernelInterface>
{
    using implementation_t = Impl;

protected:
    using signature_t = StandardKernelSignature<ArgSpec...>;

public:
    HostKernelBase();

    RPY_NO_DISCARD static std::unique_ptr<KernelArguments> new_binding()
    {
        return signature_t::make()->new_binding();
    }

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

    static Kernel get() noexcept;
};

template <typename Impl, typename... ArgSpec>
HostKernelBase<Impl, ArgSpec...>::HostKernelBase()
    : RefCountBase(signature_t::make())
{}

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
    return string(Impl::get_name());
}
template <typename Impl, typename... ArgSpec>
dimn_t HostKernelBase<Impl, ArgSpec...>::num_args() const
{
    return signature_t::num_params;
}
template <typename Impl, typename... ArgSpec>
Event HostKernelBase<Impl, ArgSpec...>::launch_kernel_async(
        Queue& queue,
        const KernelLaunchParams& params,
        const KernelArguments& args
) const
{

    using Binding = devices::ArgumentBinder<
            typename signature_t::ParamsList,
            decltype(Impl::run)>;

    Binding::eval(
            [&params](auto&&... kargs) {
                Impl::run(params, std::forward<decltype(kargs)>(kargs)...);
            },
            args
    );

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
            typename signature_t::ParamsList,
            decltype(Impl::run)>;
    Binding::eval(
            [&params](auto&&... kargs) {
                Impl::run(params, std::forward<decltype(kargs)>(kargs)...);
            },
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
    RPY_NO_DISCARD static string_view get_base_name() noexcept
    {
        static string name = string_cat("unary_", Operator<Value>::name);
        return name;
    }
    RPY_NO_DISCARD static string_view get_name() noexcept
    {
        static string name
                = string_cat(get_base_name(), '_', devices::type_id_of<T>);
        return name;
    }

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
                    const auto index
                            = (ix * group_size.y + iy) * group_size.z + iz;
                    result[index] = op(arg[index]);
                }
            }
        }
    }
};

template <template <typename> class Operator, typename T>
class UnaryInplaceHostKernel : public dtl::HostKernelBase<
                                       UnaryInplaceHostKernel<Operator, T>,
                                       params::ResultBuffer<T>,
                                       params::Operator<T>>
{
public:
    RPY_NO_DISCARD static string_view get_base_name() noexcept
    {
        static string name
                = string_cat("inplace_unary_", Operator<Value>::name);
        return name;
    }
    RPY_NO_DISCARD static string_view get_name() noexcept
    {
        static string name
                = string_cat(get_base_name(), '_', devices::type_id_of<T>);
        return name;
    }

    static void
    run(const KernelLaunchParams& params, Slice<T> buffer, Operator<T>&& op)
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
    RPY_NO_DISCARD static string_view get_base_name() noexcept
    {
        static string name = string_cat("binary_", Operator<Value>::name);
        return name;
    }
    RPY_NO_DISCARD static string_view get_name() noexcept
    {
        static string name
                = string_cat(get_base_name(), '_', devices::type_id_of<T>);
        return name;
    }

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
    RPY_NO_DISCARD static string_view get_base_name() noexcept
    {
        static string name
                = string_cat("inplace_binary_", Operator<Value>::name);
        return name;
    }

    RPY_NO_DISCARD static string_view get_name() noexcept
    {
        static string name
                = string_cat(get_base_name(), '_', devices::type_id_of<T>);
        return name;
    }

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

namespace dtl {

template <typename Impl, typename... ArgSpec>
Kernel HostKernelBase<Impl, ArgSpec...>::get() noexcept
{
    return Kernel(new Impl());
}

}// namespace dtl

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICES_HOST_KERNEL_H
