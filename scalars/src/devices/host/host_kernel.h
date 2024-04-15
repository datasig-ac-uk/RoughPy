// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

//
// Created by user on 16/10/23.
//

#ifndef ROUGHPY_DEVICE_SRC_CPUDEVICE_CPU_KERNEL_H_
#define ROUGHPY_DEVICE_SRC_CPUDEVICE_CPU_KERNEL_H_

#include "devices/buffer.h"
#include "devices/event.h"
#include "devices/kernel.h"

#include "host_event.h"

#include <exception>
#include <functional>
#include <mutex>

namespace rpy {
namespace devices {

template <typename... Ts>
using raw_kfn_ptr = void (*)(Ts...);

// Defined elsewhere :|
struct ThreadingContext;

class HostKernelEvent : public dtl::RefCountBase<EventInterface>
{
    mutable HostEventContext m_ctx;

    string m_kernel_name;

public:
    explicit HostKernelEvent(string_view kernel_name);

    Device device() const noexcept override;
    void* ptr() noexcept override;
    const void* ptr() const noexcept override;
    void wait() override;
    EventStatus status() const override;
    bool is_user() const noexcept override;
};

class CPUKernel : public dtl::RefCountBase<KernelInterface>
{

    struct LaunchContext {
        HostEventContext* event;
        ThreadingContext* threads = nullptr;
    };

    using wrapped_kernel_t = std::function<
            void(LaunchContext*,
                 const KernelLaunchParams&,
                 Slice<KernelArgument>)>;

    Event new_kernel_event(string_view name) const;

    wrapped_kernel_t m_kernel;
    string m_name;
    uint32_t m_nargs;

public:
    template <typename... Ts>
    CPUKernel(raw_kfn_ptr<Ts...> fn, string name);

    template <typename... Ts>
    CPUKernel(std::function<void(Ts...)> fn, string name);
    bool is_host() const noexcept override;
    Device device() const noexcept override;

    string name() const override;
    dimn_t num_args() const override;
    Event launch_kernel_async(
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

namespace dtl {

template <typename T>
class ConvertedKernelArgument
{
    T* p_data;

public:
    explicit ConvertedKernelArgument(KernelArgument& arg);

    operator T() { return *p_data; }
};

template <typename T>
ConvertedKernelArgument<T>::ConvertedKernelArgument(KernelArgument& arg)
{
    RPY_CHECK(arg.info() == type_info<T>());
    p_data = arg.const_pointer();
}

template <typename T>
class ConvertedKernelArgument<T*>
{
    T* p_data;
    Buffer m_view;

public:
    explicit ConvertedKernelArgument(KernelArgument& arg);

    operator T*() { return p_data; }
};

template <typename T>
ConvertedKernelArgument<T*>::ConvertedKernelArgument(KernelArgument& arg)
{
    RPY_CHECK(arg.info() == type_info<T>());
    RPY_CHECK(arg.is_buffer() && !arg.is_const());
    auto& buffer = arg.buffer();

    if (buffer.device() != get_host_device()) {
        m_view = buffer.map(buffer.size(), 0);
        p_data = reinterpret_cast<T*>(m_view.ptr());
    } else {
        p_data = buffer.ptr();
    }
}

template <typename T>
class ConvertedKernelArgument<const T*>
{
    const T* p_data;
    Buffer m_view;

public:
    explicit ConvertedKernelArgument(KernelArgument& arg);

    operator const T*() { return p_data; }
};

template <typename T>
ConvertedKernelArgument<const T*>::ConvertedKernelArgument(KernelArgument& arg)
{
    RPY_CHECK(arg.info() == type_info<T>());
    RPY_CHECK(arg.is_buffer() && !arg.is_const());
    const auto& buffer = arg.const_buffer();

    if (buffer.device() != get_host_device()) {
        m_view = buffer.map();
        p_data = reinterpret_cast<T*>(m_view.ptr());
    } else {
        p_data = buffer.ptr();
    }
}

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
 * However, this invokes undefined behaviour: the increment operators are not
 * guaranteed to happen in order. To get around this, we have to explicitly
 * index into the argument array by first making an index template pack that
 * can be unpacked at the same time to disambiguate the increment operations.
 * This is based on answers on SO: https://stackoverflow.com/a/11044592/9225581,
 * https://stackoverflow.com/a/10930078/9225581, and the referenced article
 * http://loungecpp.wikidot.com/tips-and-tricks%3aindices
 *
 */

template <size_t... Is>
struct Indices {
};

template <size_t N, size_t... Is>
struct BuildIndices {
    using type = typename BuildIndices<N - 1, N - 1, Is...>::type;
};

template <size_t... Is>
struct BuildIndices<0, Is...> {
    using type = Indices<Is...>;
};

template <typename... Ts>
using indices_for = BuildIndices<sizeof...(Ts)>;

template <typename... Ts, size_t... Is>
RPY_INLINE_ALWAYS void invoke_kernel_inner(
        std::function<void(Ts...)> fn,
        const KernelLaunchParams& params,
        Indices<Is...> RPY_UNUSED_VAR indices,
        Slice<KernelArgument> args
)
{
    fn(ConvertedKernelArgument<Ts>(args[Is])...);
}

template <typename... Ts>
RPY_INLINE_ALWAYS void invoke_kernel(
        std::function<void(Ts...)> fn,
        const KernelLaunchParams& params,
        Slice<KernelArgument> args
)
{
    invoke_kernel_inner(std::move(fn), params, indices_for<Ts...>(), args);
}

template <typename... Ts, size_t... Is>
RPY_INLINE_ALWAYS void invoke_kernel_inner(
        raw_kfn_ptr<Ts...> fn,
        const KernelLaunchParams& params,
        Indices<Is...>,
        Slice<KernelArgument> args
)
{
    fn(ConvertedKernelArgument<Ts>(args[Is])...);
}

template <typename... Ts>
RPY_INLINE_ALWAYS void invoke_kernel(
        raw_kfn_ptr<void(Ts...)> fn,
        const KernelLaunchParams& params,
        Slice<KernelArgument> args
)
{
    invoke_kernel_inner(fn, params, indices_for<Ts...>(), args);
}

template <typename F>
void invoke_locking(
        F&& fn,
        HostEventContext* event_ctx,
        ThreadingContext* RPY_UNUSED_VAR thread_ctx,
        const KernelLaunchParams& params,
        Slice<KernelArgument> args
) noexcept
{
    std::unique_lock<std::mutex> lk(event_ctx->lock);
    event_ctx->status = EventStatus::Running;
    lk.unlock();
    event_ctx->condition.notify_all();

    try {
        invoke_kernel(std::forward<F>(fn), params, args);
        lk.lock();
        event_ctx->status = EventStatus::CompletedSuccessfully;
    } catch (...) {
        lk.lock();
        event_ctx->status = EventStatus::Error;
        event_ctx->error_state = std::current_exception();
    }
    lk.unlock();
    event_ctx->condition.notify_all();
}

template <typename F>
void invoke_nonlocking(
        F&& fn,
        HostEventContext* event_ctx,
        const KernelLaunchParams& params,
        Slice<KernelArgument> args
) noexcept
{
    event_ctx->status = EventStatus::Running;
    try {
        invoke_kernel(std::forward<F>(fn), params, args);
        event_ctx->status = EventStatus::CompletedSuccessfully;
    } catch (...) {
        event_ctx->status = EventStatus::Error;
        event_ctx->error_state = std::current_exception();
    }
}

}// namespace dtl

template <typename... Ts>
CPUKernel::CPUKernel(raw_kfn_ptr<Ts...> fn, string name)
    : m_kernel([fn](LaunchContext* ctx,
                    const KernelLaunchParams& params,
                    Slice<KernelArgument> args) {
          if (ctx->threads != nullptr) {
              dtl::invoke_locking(fn, ctx->event, ctx->threads, params, args);
          } else {
              dtl::invoke_nonlocking(fn, ctx->event, params, args);
          }
      }),
      m_nargs(sizeof...(Ts)),
      m_name(std::move(name))
{}

template <typename... Ts>
CPUKernel::CPUKernel(std::function<void(Ts...)> fn, string name)
    : m_kernel([fn = std::move(fn
                )](LaunchContext* ctx,
                   const KernelLaunchParams& params,
                   Slice<KernelArgument> args) {
          if (ctx->threads != nullptr) {
              dtl::invoke_locking(fn, ctx->event, ctx->threads, params, args);
          } else {
              dtl::invoke_nonlocking(fn, ctx->event, params, args);
          }
      }),
      m_nargs(sizeof...(Ts)),
      m_name(std::move(name))
{}

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_SRC_CPUDEVICE_CPU_KERNEL_H_
