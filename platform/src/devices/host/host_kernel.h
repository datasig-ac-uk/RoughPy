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
#include "devices/core.h"
#include "devices/event.h"
#include "devices/kernel.h"
#include "devices/type.h"

#include "device_support/fundamental_type.h"
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
                 const KernelArguments&)>;

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
            const KernelArguments& args
    ) const override;
    EventStatus launch_kernel_sync(
            Queue& queue,
            const KernelLaunchParams& params,
            const KernelArguments& args
    ) const override;
};


template <typename... Ts>
CPUKernel::CPUKernel(raw_kfn_ptr<Ts...> fn, string name)
    : m_kernel([fn](LaunchContext* ctx,
                    const KernelLaunchParams& params,
                    const KernelArguments& args) {

      }),
      m_nargs(sizeof...(Ts)),
      m_name(std::move(name))
{}

template <typename... Ts>
CPUKernel::CPUKernel(std::function<void(Ts...)> fn, string name)
    : m_kernel([fn = std::move(fn
                )](LaunchContext* ctx,
                   const KernelLaunchParams& params,
                   const KernelArguments& args) {

      }),
      m_nargs(sizeof...(Ts)),
      m_name(std::move(name))
{}

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_SRC_CPUDEVICE_CPU_KERNEL_H_
