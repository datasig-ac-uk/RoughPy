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
// Created by sam on 25/08/23.
//

#ifndef ROUGHPY_OPEN_CL_DEVICE_H
#define ROUGHPY_OPEN_CL_DEVICE_H

#include <roughpy/device/core.h>
#include <roughpy/device/device_handle.h>
#include <roughpy/device/kernel.h>


#include <unordered_map>
#include <vector>

#include <roughpy/device/queue.h>

#include "opencl_headers.h"

namespace rpy {
namespace device {

class OpenCLDevice : public DeviceHandle
{
    cl_device_id m_device;
    int32_t m_device_id;

    cl_context m_ctx;

    cl_command_queue m_default_queue;

    std::vector<cl_program> m_programs;
    std::unordered_map<string, cl_kernel> m_kernels;

public:
    explicit OpenCLDevice(cl_device_id device);
    ~OpenCLDevice() override;

    optional<fs::path> runtime_library() const noexcept override;
    DeviceInfo info() const noexcept override;

    cl_command_queue default_queue() const noexcept
    {
        return m_default_queue;
    }
    cl_context context() const noexcept { return m_ctx; }

};

}// namespace device
}// namespace rpy

#endif// ROUGHPY_OPEN_CL_DEVICE_H
