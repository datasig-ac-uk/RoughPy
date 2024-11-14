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

#include <atomic>
#include <memory>

#include "devices/core.h"
#include "devices/device_object_base.h"
#include "devices/event.h"
#include "devices/buffer.h"




namespace rpy {
namespace devices {

class CPUBuffer : public dtl::RefCountBase<BufferInterface>
{
public:
    using atomic_t = std::atomic_size_t*;
private:


    enum Flags
    {
        IsConst = 1,
    };

    struct RawBuffer {
        void* ptr = nullptr;
        dimn_t size = 0;
    };

    RawBuffer raw_buffer;
    Flags flags;


    CPUBuffer(RawBuffer raw, Flags arg_flags);

public:


    CPUBuffer(void* raw_ptr, dimn_t size);
    CPUBuffer(const void* raw_ptr, dimn_t size);

    ~CPUBuffer() override;

    std::unique_ptr<dtl::InterfaceBase> clone() const override;
    Device device() const noexcept override;

    BufferMode mode() const override;
    dimn_t size() const override;
    void* ptr() noexcept override;
    DeviceType type() const noexcept override;
    const void* ptr() const noexcept override;

    Event
    to_device(Buffer& dst, const Device& device, Queue& queue) override;

    void* map(BufferMode map_mode, dimn_t size, dimn_t offset) override;
};

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_SRC_CPUDEVICE_CPU_BUFFER_H_
