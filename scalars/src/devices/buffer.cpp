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
// Created by user on 11/10/23.
//

#include "devices/buffer.h"
#include "devices/device_handle.h"

#include "devices/event.h"
#include "devices/kernel.h"
#include "devices/queue.h"

#include "devices/host/host_buffer.h"

using namespace rpy;
using namespace rpy::devices;

namespace rpy {
namespace devices {
namespace dtl {

template class RPY_DLL_EXPORT ObjectBase<BufferInterface, Buffer>;
}
}// namespace devices
}// namespace rpy

Buffer::Buffer(void* ptr, dimn_t size, TypeInfo info)
    : base_t(new CPUBuffer(ptr, size, info))
{}

Buffer::Buffer(const void* ptr, dimn_t size, TypeInfo info)
    : base_t(new CPUBuffer(ptr, size, info))
{}

BufferMode Buffer::mode() const
{
    if (!impl()) { return BufferMode::Read; }
    return impl()->mode();
}

TypeInfo BufferInterface::type_info() const noexcept
{
    return devices::type_info<char>();
}
dimn_t Buffer::size() const
{
    if (!impl()) { return 0; }
    return impl()->size();
}

static inline bool check_device_compatibility(Buffer& dst, const Device& device)
{
    if (dst.is_null() || !device) { return true; }

    RPY_CHECK(dst.device() == device);

    return true;
}

void Buffer::to_device(Buffer& dst, const Device& device)
{
    if (impl() && check_device_compatibility(dst, device)) {
        auto queue = device->get_default_queue();
        impl()->to_device(dst, device, queue).wait();
    }
}
Event Buffer::to_device(Buffer& dst, const Device& device, Queue& queue)
{
    if (impl() && check_device_compatibility(dst, device)) {
        return impl()->to_device(dst, device, queue);
    }
    return {};
}

Buffer Buffer::map(rpy::dimn_t size, rpy::dimn_t offset)
{
    if (impl() == nullptr) { return Buffer(); }

    RPY_DBG_ASSERT(!is_null());
    auto max_size = this->size();
    RPY_CHECK(offset + size <= max_size);

    auto info = this->type_info();

    // The "owner" might itself be borrowed, by calling memory_owner we will
    // always get the "real" owning buffer for this memory
    auto memory_owner = this->memory_owner();

    // owner might not be a CPU buffer, so this might be a real memory mapping.
    auto* ptr = impl()->map(size, offset);

    return Buffer(new CPUBuffer(ptr, size, info));
}

Buffer Buffer::map(rpy::dimn_t size, rpy::dimn_t offset) const
{
    if (impl() == nullptr) { return Buffer(); }

    RPY_DBG_ASSERT(!is_null());
    auto max_size = this->size();
    RPY_CHECK(offset + size <= max_size);

    auto info = this->type_info();

    // The "owner" might itself be borrowed, by calling memory_owner we will
    // always get the "real" owning buffer for this memory
    auto memory_owner = this->memory_owner();

    // owner might not be a CPU buffer, so this might be a real memory mapping.
    const auto* ptr = impl()->map(size, offset);

    return Buffer(new CPUBuffer(ptr, size, info));
}

TypeInfo Buffer::type_info() const noexcept
{
    if (!impl()) { return rpy::devices::type_info<char>(); }
    return impl()->type_info();
}

Buffer Buffer::memory_owner() const noexcept
{
    if (impl() == nullptr) { return Buffer(); }
    return impl()->memory_owner();
}
