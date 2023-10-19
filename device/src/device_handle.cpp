// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
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
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//
// Created by user on 11/10/23.
//

#include <roughpy/device/device_handle.h>

#include <roughpy/device/buffer.h>

#include <mutex>
#include <vector>

using namespace rpy;
using namespace rpy::devices;

DeviceHandle::DeviceHandle() {}
DeviceInfo DeviceHandle::info() const noexcept { return {}; }
optional<fs::path> DeviceHandle::runtime_library() const noexcept
{
    return {};
}

DeviceHandle::~DeviceHandle() = default;

Buffer DeviceHandle::raw_alloc(rpy::dimn_t count, rpy::dimn_t alignment) const
{
    return {};
}

void DeviceHandle::raw_free(void* pointer, dimn_t size) const {}

const Kernel& DeviceHandle::register_kernel(Kernel kernel) const {
    RPY_CHECK(kernel.device() == this);
    const guard_type access(get_lock());

    string name = kernel.name();
    auto& cached = m_kernel_cache[name];
    if (cached.is_nop()) {
        cached = std::move(kernel);
    }

    return cached;
}

optional<Kernel> DeviceHandle::get_kernel(const string& name) const noexcept
{
    const guard_type access(get_lock());

    const auto found = m_kernel_cache.find(name);
    if (found != m_kernel_cache.end()) {
        return found->second;
    }
    return {};
}

optional<Kernel> DeviceHandle::compile_kernel_from_str(string_view code) const
{
    return {};
}
void DeviceHandle::compile_kernels_from_src(string_view RPY_UNUSED_VAR code
) const
{}
Event DeviceHandle::new_event() const { return {}; }
Queue DeviceHandle::new_queue() const { return {}; }
Queue DeviceHandle::get_default_queue() const { return {}; }

optional<boost::uuids::uuid> DeviceHandle::uuid() const noexcept
{
    return {};
}
optional<PCIBusInfo> DeviceHandle::pci_bus_info() const noexcept
{
    return {};
}
bool DeviceHandle::supports_type(const TypeInfo& info) const noexcept
{
    return false;
}
