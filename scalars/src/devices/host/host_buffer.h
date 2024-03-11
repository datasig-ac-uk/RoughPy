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

#ifndef ROUGHPY_DEVICE_SRC_CPUDEVICE_CPU_BUFFER_H_
#define ROUGHPY_DEVICE_SRC_CPUDEVICE_CPU_BUFFER_H_

#include "devices/buffer.h"

#include "devices/core.h"
#include "devices/event.h"
#include "devices/queue.h"
#include "host_decls.h"

#include <roughpy/core/types.h>

#include <atomic>

namespace rpy {
namespace devices {

class CPUBuffer : public dtl::RefCountBase<BufferInterface>
{

    enum Flags
    {
        IsConst = 1,
        IsOwned = 2,
    };

    RawBuffer raw_buffer;
    uint32_t m_flags;
    TypeInfo m_info;
    Buffer m_memory_owner;

    CPUBuffer(RawBuffer raw, uint32_t arg_flags, TypeInfo info);

public:
    CPUBuffer(dimn_t size, TypeInfo info);
    CPUBuffer(void* raw_ptr, dimn_t size, TypeInfo info);
    CPUBuffer(const void* raw_ptr, dimn_t size, TypeInfo info);

    ~CPUBuffer() override;

    Device device() const noexcept override;

    bool is_host() const noexcept override;
    BufferMode mode() const override;
    TypeInfo type_info() const noexcept override;
    dimn_t size() const override;
    void* ptr() noexcept override;
    DeviceType type() const noexcept override;
    const void* ptr() const noexcept override;

    Event
    to_device(Buffer& dst, const Device& device, Queue& queue) override;

    Buffer map_mut(dimn_t size, dimn_t offset) override;
    Buffer map(dimn_t size, dimn_t offset) const override;

    Buffer memory_owner() const noexcept override;

    Buffer slice(dimn_t offset, dimn_t size) const override;
    Buffer mut_slice(dimn_t offset, dimn_t size) override;
    void unmap(Buffer& ptr) const noexcept override;
    dimn_t bytes() const override;
};

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_SRC_CPUDEVICE_CPU_BUFFER_H_
