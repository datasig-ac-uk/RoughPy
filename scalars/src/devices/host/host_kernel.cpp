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

#include "host_kernel.h"
#include "host_device_impl.h"

#include "host_event.h"

#include "devices/event.h"

#include <roughpy/core/helpers.h>

#include <memory>
#include <mutex>

using namespace rpy;
using namespace rpy::devices;

string CPUKernel::name() const { return m_name; }
dimn_t CPUKernel::num_args() const { return m_nargs; }

Event CPUKernel::new_kernel_event(string_view name)
{
    auto inner = std::make_unique<HostKernelEvent>(name);
    return Event(inner.release());
}
Event CPUKernel::launch_kernel_async(
        Queue& queue,
        const KernelLaunchParams& params,
        Slice<KernelArgument> args
)
{
    auto event = new_kernel_event(m_name);
    LaunchContext ctx{
            reinterpret_cast<HostEventContext*>(event.ptr()),
            nullptr
    };
    m_kernel(&ctx, params, args);
    return event;
}
EventStatus CPUKernel::launch_kernel_sync(
        rpy::devices::Queue& queue,
        const rpy::devices::KernelLaunchParams& params,
        Slice<rpy::devices::KernelArgument> args
)
{
    auto event = new_kernel_event(m_name);
    LaunchContext ctx{
            reinterpret_cast<HostEventContext*>(event.ptr()),
            nullptr
    };
    m_kernel(&ctx, params, args);
    return event.status();
}

Device CPUKernel::device() const noexcept { return get_host_device(); }
bool CPUKernel::is_host() const noexcept { return true; }

HostKernelEvent::HostKernelEvent(std::string_view kernel_name)
    : m_ctx{std::mutex(),
            std::condition_variable(),
            EventStatus::Queued,
            nullptr},
      m_kernel_name(kernel_name)
{}

void HostKernelEvent::wait()
{
    std::unique_lock<std::mutex> lk(m_ctx.lock);
    m_ctx.condition.wait(lk, [this] {
        return m_ctx.status == EventStatus::CompletedSuccessfully
                || m_ctx.status == EventStatus::Error;
    });
}
EventStatus HostKernelEvent::status() const
{
    const std::lock_guard<std::mutex> access(m_ctx.lock);
    return m_ctx.status;
}
bool HostKernelEvent::is_user() const noexcept { return false; }
Device HostKernelEvent::device() const noexcept { return get_host_device(); }
void* HostKernelEvent::ptr() noexcept { return &m_ctx; }
const void* HostKernelEvent::ptr() const noexcept { return &m_ctx; }
