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

#include "ocl_buffer.h"

#include "opencl_headers.h"

using namespace rpy;
using namespace rpy::device;

#define CL_MEM_MODE_MASK \
    (CL_MEM_READ_ONLY | CL_MEM_WRITE_ONLY | CL_MEM_READ_WRITE)



BufferMode OCLBufferInterface::mode(void* content) const
{
    cl_int raw_mode;
    auto ecode = ::clGetMemObjectInfo(
            buffer(content),
            CL_MEM_FLAGS,
            sizeof(cl_int),
            &raw_mode,
            nullptr
            );
    RPY_CHECK(ecode == CL_SUCCESS);

    switch ( raw_mode & CL_MEM_MODE_MASK) {
        case CL_MEM_READ_ONLY:
            return BufferMode::Read;
        case CL_MEM_WRITE_ONLY:
            return BufferMode::Write;
        case CL_MEM_READ_WRITE:
            return BufferMode::ReadWrite;
        default:
            RPY_THROW(std::runtime_error, "invalid mode for memory buffer");
    }
}
dimn_t OCLBufferInterface::size(void* content) const
{
    dimn_t raw_size;
    auto ecode = ::clGetMemObjectInfo(
            buffer(content),
            CL_MEM_SIZE,
            sizeof(raw_size),
            &raw_size,
            nullptr
            );
    RPY_CHECK(ecode == CL_SUCCESS);
    return raw_size;
}
void* OCLBufferInterface::ptr(void* content) const
{
    return const_cast<void*>(static_cast<const void*>(this));
}
void* OCLBufferInterface::clone(void* content) const
{
    dimn_t buf_size = size(content);

    cl_int ecode;
    cl_mem new_buffer = ::clCreateBuffer(
            device(content)->context(),
            CL_MEM_ALLOC_HOST_PTR,
            buf_size,
            nullptr,
            &ecode
            );
    RPY_CHECK(new_buffer != nullptr);

    cl_event event;
    ecode = ::clEnqueueCopyBuffer(
            device(content)->default_queue(),
            buffer(content),
            new_buffer,
            0,
            0,
            buf_size,
            0,
            nullptr,
            &event
            );
    RPY_CHECK(ecode == CL_SUCCESS);

    ::clWaitForEvents(1, &event);

    return create_data(new_buffer, device(content));
}
void OCLBufferInterface::clear(void* content) const
{
    auto ret = ::clReleaseMemObject(buffer(content));
    RPY_CHECK(ret == CL_SUCCESS);
    delete static_cast<Data*>(content);
}

const OCLBufferInterface* cl::buffer_interface() noexcept {
    static const OCLBufferInterface iface;
    return &iface;
}



