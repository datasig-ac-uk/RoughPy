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

#ifndef ROUGHPY_DEVICE_SRC_OPENCL_OCL_BUFFER_H_
#define ROUGHPY_DEVICE_SRC_OPENCL_OCL_BUFFER_H_

#include "devices/buffer.h"

#include "ocl_decls.h"
#include "ocl_headers.h"

namespace rpy {
namespace devices {


class OCLBuffer : public BufferInterface
{
    OCLDevice m_device;
    cl_mem m_buffer;

public:

    OCLBuffer(cl_mem buffer, OCLDevice dev) noexcept;
    ~OCLBuffer() override;

    RPY_NO_DISCARD
    DeviceType type() const noexcept override;
    RPY_NO_DISCARD
    dimn_t ref_count() const noexcept override;
    RPY_NO_DISCARD
    std::unique_ptr<dtl::InterfaceBase> clone() const override;
    RPY_NO_DISCARD
    Device device() const noexcept override;
    RPY_NO_DISCARD
    BufferMode mode() const override;
    RPY_NO_DISCARD
    dimn_t size() const override;
    RPY_NO_DISCARD
    void* ptr() noexcept override;
    const void* ptr() const noexcept override;

    Event
    to_device(Buffer& dst, const Device& device, Queue& queue) override;

    reference_count_type inc_ref() noexcept override;
    reference_count_type dec_ref() noexcept override;

    void* map(BufferMode map_mode, dimn_t size, dimn_t offset) override;
    void unmap(void* ptr) noexcept override;
};

}// namespace device
}// namespace rpy

#endif// ROUGHPY_DEVICE_SRC_OPENCL_OCL_BUFFER_H_
