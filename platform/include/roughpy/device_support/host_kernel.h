//
// Created by sam on 30/04/24.
//

#ifndef ROUGHPY_DEVICES_HOST_KERNEL_H
#define ROUGHPY_DEVICES_HOST_KERNEL_H

#include <roughpy/core/traits.h>
#include <roughpy/devices/host_device.h>
#include <roughpy/devices/kernel.h>
#include <roughpy/devices/kernel_arg.h>

#include <boost/compressed_pair.hpp>

namespace rpy {
namespace devices {

/**
 * @brief The HostKernel class represents a host kernel.
 *
 * The HostKernel class is a concrete class that implements the KernelInterface
 * interface. It allows for executing a host kernel.
 */
template <typename F>
class HostKernel : public dtl::RefCountBase<KernelInterface>
{
    boost::compressed_pair<F, string> m_func_and_name;

public:
    HostKernel(F&& func, string name)
        : m_func_and_name(std::move(func), std::move(name))
    {}

    explicit HostKernel(string name)
        : m_func_and_name(F(), std::move(name))
    {}

    RPY_NO_DISCARD bool is_host() const noexcept override;
    RPY_NO_DISCARD DeviceType type() const noexcept override;
    RPY_NO_DISCARD Device device() const noexcept override;
    RPY_NO_DISCARD string name() const override;
    RPY_NO_DISCARD dimn_t num_args() const override;
    RPY_NO_DISCARD Event launch_kernel_async(
            Queue& queue,
            const KernelLaunchParams& params,
            Slice<KernelArgument> args
    ) const override;
    EventStatus launch_kernel_sync(
            Queue& queue,
            const KernelLaunchParams& params,
            Slice<KernelArgument> args
    ) const override;
};

template <typename F>
bool HostKernel<F>::is_host() const noexcept
{
    return true;
}
template <typename F>
DeviceType HostKernel<F>::type() const noexcept
{
    return DeviceType::CPU;
}
template <typename F>
Device HostKernel<F>::device() const noexcept
{
    return get_host_device();
}
template <typename F>
string HostKernel<F>::name() const
{
    return m_func_and_name.second();
}
template <typename F>
dimn_t HostKernel<F>::num_args() const
{
    return rpy::num_args<F>;
}

namespace dtl {

template <typename T>
class ConvertedKernelArgument
{
    T* p_data;

public:
    explicit ConvertedKernelArgument(KernelArgument& arg)
    {
        RPY_CHECK(arg.info() == type_info<T>());
        p_data = arg.const_pointer();
    }

    RPY_NO_DISCARD operator T() { return *p_data; }
};

template <typename T>
class ConvertedKernelArgument<T&>
{
    T* p_data;

public:
    explicit ConvertedKernelArgument(KernelArgument& arg)
    {
        RPY_CHECK(arg.info() == type_info<T>());
        p_data = static_cast<T*>(arg.pointer());
    }

    RPY_NO_DISCARD operator T&() { return *p_data; }
};

template <typename T>
class ConvertedKernelArgument<const T&>
{
     const T* p_data;

public:
    explicit ConvertedKernelArgument(KernelArgument& arg)
    {
        RPY_CHECK(arg.info() == type_info<T>());
        p_data = static_cast<const T*>(arg.const_pointer());
    }

    RPY_NO_DISCARD operator const T&() { return *p_data; }
};


template <typename T>
class ConvertedKernelArgument<Slice<T>>
{
    Slice<T> m_slice;
    Buffer m_view;

public:
    explicit ConvertedKernelArgument(KernelArgument& arg)
    {
        RPY_CHECK(arg.info() == type_info<T>());
        RPY_CHECK(arg.is_buffer() && !arg.is_const());
        auto& buffer = arg.buffer();

        if (buffer.is_host()) {
            m_view = buffer.map(buffer.size(), 0);
        } else {
            m_view = buffer;
        }

        m_slice = m_view.as_mut_slice<T>();
    }

    RPY_NO_DISCARD operator Slice<T>() const noexcept { return m_slice; }
};

template <typename T>
class ConvertedKernelArgument<Slice<const T>>
{
    Slice<const T> m_slice;
    Buffer m_view;

public:
    explicit ConvertedKernelArgument(KernelArgument& arg)
    {
        RPY_CHECK(arg.info() == type_info<T>());
        RPY_CHECK(arg.is_buffer() && !arg.is_const());
        const auto& buffer = arg.const_buffer();

        if (buffer.is_host()) {
            m_view = buffer.map(buffer.size(), 0);
        } else {
            m_view = buffer;
        }

        m_slice = m_view.as_slice<const T>();
    }

    RPY_NO_DISCARD operator Slice<const T>() const noexcept { return m_slice; }
};
/*
 * The problem we need to solve now is how to invoke a function that takes a
 * normal list of arguments, given an array of KernelArguments. To do this,
 * we're going to use the power of variadic templates to do the unpacking.
 *
 * We'd like to do something very simple like the following
 *
 * template <typename... Args>
 * void invoke(void (*)(Args... args) fn, Slice<KernelArgument> args, ...) {
 *      auto* aptr = args.data();
 *      fn(cast<Args>(aptr++)...);
 * }
 *
 * However, this invokes undefined behaviour: the increment operators are
 * not guaranteed to happen in order. To get around this, we have to
 * explicitly index into the argument array by first making an index
 * template pack that can be unpacked at the same time to disambiguate the
 * increment operations. This is based on answers on SO:
 * https://stackoverflow.com/a/11044592/9225581,
 * https://stackoverflow.com/a/10930078/9225581, and the referenced article
 * http://loungecpp.wikidot.com/tips-and-tricks%3aindices
 *
 */

template <typename F, dimn_t... Is>
RPY_INLINE_ALWAYS void invoke_kernel_inner(
        F&& kernel_fn,
        const KernelLaunchParams& RPY_UNUSED_VAR params,
        integer_sequence<dimn_t, Is...> RPY_UNUSED_VAR indices,
        Slice<KernelArgument> args
)
{
    kernel_fn(ConvertedKernelArgument<arg_at_t<F, Is>>(args[Is])...);
}

template <typename F>
RPY_INLINE_ALWAYS void invoke_kernel(
        F&& kernel_fn,
        const KernelLaunchParams& params,
        Slice<KernelArgument> args
)
{
    using seq = make_integer_sequence<dimn_t, num_args<F>>;
    invoke_kernel_inner(std::forward<F>(kernel_fn), params, seq(), args);
}

}// namespace dtl

template <typename F>
Event HostKernel<F>::launch_kernel_async(
        Queue& queue,
        const KernelLaunchParams& params,
        Slice<KernelArgument> args
) const
{
    dtl::invoke_kernel(m_func_and_name.first(), params, args);
    return Event();
}
template <typename F>
EventStatus HostKernel<F>::launch_kernel_sync(
        Queue& queue,
        const KernelLaunchParams& params,
        Slice<KernelArgument> args
) const
{
    dtl::invoke_kernel(m_func_and_name.first(), params, args);
    return EventStatus::CompletedSuccessfully;
}

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICES_HOST_KERNEL_H
