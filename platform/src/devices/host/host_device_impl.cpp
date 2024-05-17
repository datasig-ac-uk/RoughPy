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

#include "host_device_impl.h"

#include <roughpy/core/alloc.h>
#include <roughpy/core/container/map.h>

#include "devices/buffer.h"
#include "devices/event.h"
#include "devices/kernel.h"
#include "devices/queue.h"

#include "host_buffer.h"
#include "host_event.h"
#include "host_kernel.h"
#include "host_queue.h"

// #include "kernels/masked_binary.h"
// #include "kernels/masked_unary.h"

#include <algorithm>

using namespace rpy;
using namespace rpy::devices;

CPUDeviceHandle::CPUDeviceHandle() {}
CPUDeviceHandle::~CPUDeviceHandle() = default;

CPUDevice CPUDeviceHandle::get()
{
    static Rc<const CPUDeviceHandle> device(new CPUDeviceHandle);
    return device;
}

bool CPUDeviceHandle::is_host() const noexcept { return true; }

DeviceInfo CPUDeviceHandle::info() const noexcept
{
    return {DeviceType::CPU, 0};
}

RawBuffer CPUDeviceHandle::allocate_raw_buffer(
        rpy::dimn_t size,
        rpy::dimn_t alignment
) const
{
    RPY_DBG_ASSERT(size > 0);
    if (alignment == 0) { alignment = alignof(std::max_align_t); }

    return {aligned_alloc(alignment, size), size};
}

void CPUDeviceHandle::free_raw_buffer(rpy::devices::RawBuffer& buffer) const
{
    if (buffer.ptr != nullptr) {
        aligned_free(buffer.ptr);
        buffer.ptr = nullptr;
    }
    buffer.size = 0;
}

template <typename... Args>
Kernel make_kernel(void (*fn)(Args...)) noexcept
{
    (void) fn;
    return Kernel();
}

static const containers::FlatMap<string_view, Kernel> s_kernels{
        //        {"masked_uminus_double",
        //         make_kernel(kernels::masked_binary_into_buffer<
        //         double, std::plus<double>>)}
};

optional<Kernel> CPUDeviceHandle::get_kernel(const string& name) const noexcept
{
    auto kernel = DeviceHandle::get_kernel(name);
    if (kernel) { return kernel; }

    auto found = s_kernels.find(name);
    if (found != s_kernels.end()) { return {found->second}; }

    return {};
}
optional<Kernel>
CPUDeviceHandle::compile_kernel_from_str(const ExtensionSourceAndOptions& args
) const
{
    RPY_THROW(std::runtime_error, "no compiler available for host");
}
void CPUDeviceHandle::compile_kernels_from_src(
        const ExtensionSourceAndOptions& args
) const
{
    RPY_THROW(std::runtime_error, "no compiler available for host");
}
Event CPUDeviceHandle::new_event() const { return Event(new CPUEvent); }
Queue CPUDeviceHandle::new_queue() const { return Queue(); }
Queue CPUDeviceHandle::get_default_queue() const { return Queue(); }
bool CPUDeviceHandle::supports_type(const Type* type) const noexcept
{
    return true;
}
DeviceCategory CPUDeviceHandle::category() const noexcept
{
    return DeviceCategory::CPU;
}
bool CPUDeviceHandle::has_compiler() const noexcept { return false; }
Device CPUDeviceHandle::compute_delegate() const
{
    RPY_THROW(
            std::runtime_error,
            "no compute delegate is available on "
            "the host device"
    );
}

Buffer CPUDeviceHandle::alloc(const Type* type, dimn_t count) const
{
    if (count == 0) { return Buffer(type, 0); }

    return Buffer(new CPUBuffer(type, count));
}

Buffer CPUDeviceHandle::alloc(TypeInfo info, dimn_t count) const
{
    if (count == 0) { return Buffer(); }

    // This is a bit circular, since CPUBuffer will actually call
    // allocate_raw_buffer as part of it's own allocation, but we don't want to
    // necessarily expose the internals of CPUBuffer to make it externally
    // constructible.
    return Buffer(new CPUBuffer(count, info));
}

void CPUDeviceHandle::raw_free(Buffer& buf) const
{
    RPY_CHECK(buf.device() == this);
    buf.~Buffer();
}
