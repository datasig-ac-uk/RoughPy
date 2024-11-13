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
// Created by user on 16/10/23.
//

#include "host_buffer.h"

#include <roughpy/core/check.h>
#include <roughpy/core/debug_assertion.h>

#include "devices/device_handle.h"
#include "devices/device_object_base.h"
#include "devices/host_device.h"
#include "devices/opencl/ocl_device.h"
#include "devices/opencl/ocl_handle_errors.h"

#include "host_device_impl.h"

using namespace rpy;
using namespace rpy::devices;

CPUBuffer::CPUBuffer(CPUBuffer::RawBuffer raw, CPUBuffer::Flags arg_flags)
    : raw_buffer(std::move(raw)),
      flags(arg_flags)
{}

CPUBuffer::CPUBuffer(void* raw_ptr, dimn_t size)
    : raw_buffer{raw_ptr, size},
      flags()
{}

CPUBuffer::CPUBuffer(const void* raw_ptr, dimn_t size)
    : raw_buffer{const_cast<void*>(raw_ptr), size},
      flags(IsConst)
{}

CPUBuffer::~CPUBuffer()
{
    if (raw_buffer.ptr != nullptr) {
        RPY_DBG_ASSERT(raw_buffer.size != 0);
        get_host_device()->raw_free(raw_buffer.ptr, raw_buffer.size);
        raw_buffer = { nullptr, 0};
    } else
    {
        RPY_DBG_ASSERT(raw_buffer.size == 0);
    }
}

BufferMode CPUBuffer::mode() const
{
    if (flags & IsConst) { return BufferMode::Read; }
    return BufferMode::ReadWrite;
}
dimn_t CPUBuffer::size() const { return raw_buffer.size; }
void* CPUBuffer::ptr() noexcept { return raw_buffer.ptr; }
std::unique_ptr<rpy::devices::dtl::InterfaceBase> CPUBuffer::clone() const
{
    return std::unique_ptr<CPUBuffer>(new CPUBuffer(raw_buffer, flags));
}
Device CPUBuffer::device() const noexcept { return CPUDeviceHandle::get(); }

DeviceType CPUBuffer::type() const noexcept { return DeviceType::CPU; }
const void* CPUBuffer::ptr() const noexcept { return raw_buffer.ptr; }
Event CPUBuffer::to_device(Buffer& dst, const Device& device, Queue& queue)
{
    if (device == this->device()) {
        /*
         * There are two cases we need to work with here. The first is that
         * the buffer dst is empty, in which case we simply allocate and copy.
         * The second case is that dst has an existing allocation that can be
         * either large enough or not. If it is exactly the right size, use
         * the buffer and otherwise reallocate. Finally, copy.
         */
        if (dst.is_null()) {
            dst = device->raw_alloc(raw_buffer.size, 0);
        } else if (dst.size() != raw_buffer.size) {
            dst = device->raw_alloc(raw_buffer.size, 0);
        }

        std::memcpy(dst.ptr(), raw_buffer.ptr, raw_buffer.size);

        return {};
    }

    /*
     * If the target device is not the host device, then we need to invoke
     * the to_device function on device to initiate the copy.
     */
    return device->from_host(dst, *this, queue);
}
void* CPUBuffer::map(BufferMode map_mode, dimn_t size, dimn_t offset)
{
    if (offset + size >= raw_buffer.size) {
        RPY_THROW(
                std::invalid_argument,
                "the requested region would exceed "
                "the allocated memory of this "
                "buffer"
        );
    }

    return static_cast<byte*>(raw_buffer.ptr) + offset;
}
