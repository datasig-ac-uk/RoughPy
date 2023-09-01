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
// Created by user on 31/08/23.
//

#ifndef ROUGHPY_DEVICE_SRC_OPENCL_OCL_BUFFER_H_
#define ROUGHPY_DEVICE_SRC_OPENCL_OCL_BUFFER_H_

#include <roughpy/device/buffer.h>

#include "open_cl_runtime_library.h"
#include "open_cl_device.h"

namespace rpy {
namespace device {

class OCLBufferInterface : public BufferInterface
{
    const OpenCLRuntimeLibrary* p_runtime;

    struct Data {
        cl_mem buffer;
        boost::intrusive_ptr<OpenCLDevice> device;
    };

public:

    static void* create_data(cl_mem buffer,
                             boost::intrusive_ptr<OpenCLDevice> device)
            noexcept {
        return new Data { buffer, std::move(device) };
    }

private:

    static inline cl_mem buffer(void* content) noexcept {
        return static_cast<Data*>(content)->buffer;
    }

    static inline boost::intrusive_ptr<OpenCLDevice>& device(void* content)
            noexcept
    {
        return static_cast<Data*>(content)->device;
    }

public:

    explicit OCLBufferInterface(const OpenCLRuntimeLibrary* runtime);

    BufferMode mode(void* content) const override;
    dimn_t size(void* content) const override;
    void* ptr(void* content) const override;
    void* clone(void* content) const override;
    void clear(void* content) const override;
};

}// namespace device
}// namespace rpy

#endif// ROUGHPY_DEVICE_SRC_OPENCL_OCL_BUFFER_H_
