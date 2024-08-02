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

#include "host_buffer.h"

#include "devices/device_handle.h"
#include "devices/device_object_base.h"
#include "devices/host_device.h"

#include "devices/type.h"
#include "host_device_impl.h"

using namespace rpy;
using namespace rpy::devices;

CPUBuffer::CPUBuffer(TypePtr tp, dimn_t size)
    : base_t(std::move(tp)),
      raw_buffer{nullptr, 0},
      m_flags(IsOwned),
      m_info(type()->type_info()),
      m_num_elts(size)
{
    if (size > 0) {
        const auto& device = CPUDeviceHandle::get();
        raw_buffer = device->allocate_raw_buffer(
                size * m_info.bytes,
                m_info.alignment
        );
    }
}

CPUBuffer::CPUBuffer(TypePtr tp, void* ptr, dimn_t size)
    : base_t(tp),
      raw_buffer{ptr, size * tp->type_info().bytes},
      m_flags(IsConst),
      m_info(tp->type_info()),
      m_num_elts(size)
{}
CPUBuffer::CPUBuffer(TypePtr tp, const void* ptr, dimn_t size)
    : base_t(tp),
      raw_buffer{const_cast<void*>(ptr), size * tp->type_info().bytes},
      m_flags(IsConst),
      m_info(tp->type_info()),
      m_num_elts(size)
{}

CPUBuffer::CPUBuffer(RawBuffer raw, uint32_t arg_flags, TypeInfo info)
    : base_t(get_type(info)),
      raw_buffer(std::move(raw)),
      m_flags(arg_flags),
      m_info(info),
      m_num_elts(raw.size / info.bytes)
{
    RPY_DBG_ASSERT(raw.size == m_num_elts * info.bytes);
}

CPUBuffer::CPUBuffer(rpy::dimn_t size, rpy::devices::TypeInfo info)
    : base_t(get_type(info)),
      raw_buffer{nullptr, 0},
      m_flags(IsOwned),
      m_info(info),
      m_num_elts(size)
{
    if (size > 0) {
        auto device = CPUDeviceHandle::get();
        raw_buffer = device->allocate_raw_buffer(
                size * info.bytes,
                info.alignment
        );
    }
}

CPUBuffer::CPUBuffer(void* raw_ptr, dimn_t size, TypeInfo info)
    : base_t(get_type(info)),
      raw_buffer{raw_ptr, size * info.bytes},
      m_flags(IsNotConst),
      m_info(info),
      m_num_elts(size)
{}

CPUBuffer::CPUBuffer(dimn_t size, TypePtr type)
    : base_t(std::move(type)),
      raw_buffer{nullptr, 0},
      m_flags(IsOwned),
      m_info(type->type_info()),
      m_num_elts(size)
{
    if (size > 0) {
        auto device = CPUDeviceHandle::get();
        raw_buffer = device->allocate_raw_buffer(
                size * m_info.bytes,
                m_info.alignment
        );
    }
}
CPUBuffer::CPUBuffer(void* raw_ptr, dimn_t size, TypePtr type)
    : base_t(std::move(type)),
      raw_buffer{raw_ptr, size * type->type_info().bytes},
      m_flags(IsNotConst),
      m_info(type->type_info()),
      m_num_elts(size)
{}
CPUBuffer::CPUBuffer(const void* raw_ptr, dimn_t size, TypePtr tp)
    : base_t(std::move(tp)),
      raw_buffer{const_cast<void*>(raw_ptr), size * type()->type_info().bytes},
      m_flags(IsConst),
      m_info(type()->type_info()),
      m_num_elts(size)
{}

CPUBuffer::CPUBuffer(const void* raw_ptr, dimn_t size, TypeInfo info)
    : base_t(get_type(info)),
      raw_buffer{const_cast<void*>(raw_ptr), size * info.bytes},
      m_flags(IsConst),
      m_info(info),
      m_num_elts(size)
{}

CPUBuffer::~CPUBuffer()
{
    RPY_DBG_ASSERT(CPUBuffer::ref_count() == 0);
    if ((m_flags & IsOwned) != 0) {
        CPUDeviceHandle::get()->free_raw_buffer(raw_buffer);
    }
}
dimn_t CPUBuffer::bytes() const { return raw_buffer.size; }

BufferMode CPUBuffer::mode() const
{
    if (m_flags & IsConst) { return BufferMode::Read; }
    return BufferMode::ReadWrite;
}
void CPUBuffer::unmap(BufferInterface& other) const noexcept
{
    RPY_CHECK(other.is_host());
    RPY_CHECK(&other.memory_owner().get() == this);

    auto& as_cpubuf = static_cast<CPUBuffer&>(other);

    // Should reduce the reference count of this
    // In practice this function will probably only be called in the destructor
    // of a buffer, anyway, but better safe than sorry.
    as_cpubuf.m_memory_owner = Buffer();
    as_cpubuf.raw_buffer = {nullptr, 0};
}
dimn_t CPUBuffer::size() const { return m_num_elts; }
void* CPUBuffer::ptr() noexcept { return raw_buffer.ptr; }
Device CPUBuffer::device() const noexcept { return CPUDeviceHandle::get(); }

DeviceType CPUBuffer::device_type() const noexcept { return DeviceType::CPU; }
const void* CPUBuffer::ptr() const noexcept { return raw_buffer.ptr; }
Event CPUBuffer::to_device(Buffer& dst, const Device& device, Queue& queue)
        const
{
    if (device == this->device()) {
        /*
         * There are two cases we need to work with here. The first is that
         * the buffer dst is empty, in which case we simply allocate and copy.
         * The second case is that dst has an existing allocation that can be
         * either large enough or not. If it is exactly the right size, use
         * the buffer and otherwise reallocate. Finally, copy.
         */
        if (dst.is_null() || dst.device_type() != device_type()
            || dst.size() != raw_buffer.size) {
            dst = device->alloc(m_info, raw_buffer.size);
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

Buffer CPUBuffer::memory_owner() const noexcept
{
    if (m_memory_owner.is_null()) { return BufferInterface::memory_owner(); }
    return m_memory_owner;
}
Buffer CPUBuffer::map_mut(dimn_t size, dimn_t offset)
{
    return mut_slice(size, offset);
}
Buffer CPUBuffer::map(dimn_t size, dimn_t offset) const
{
    return slice(size, offset);
}

bool CPUBuffer::is_host() const noexcept { return true; }
Buffer CPUBuffer::slice(dimn_t size, dimn_t offset) const
{
    RPY_CHECK(offset + size <= m_num_elts);

    const auto* ptr
            = static_cast<const byte*>(raw_buffer.ptr) + offset * m_info.bytes;
    return Buffer(new CPUBuffer(ptr, size, type()));
}
Buffer CPUBuffer::mut_slice(dimn_t size, dimn_t offset)
{
    RPY_CHECK(offset + size <= m_num_elts);

    auto* ptr = static_cast<byte*>(raw_buffer.ptr) + offset * m_info.bytes;
    return Buffer(new CPUBuffer(ptr, size, type()));
}
