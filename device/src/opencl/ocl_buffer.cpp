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

#include "ocl_buffer.h"

#include "ocl_device.h"
#include "ocl_handle_errors.h"
#include "ocl_headers.h"

#include <utility>

using namespace rpy;
using namespace rpy::device;

constexpr cl_int MODE_MASK
        = CL_MEM_READ_ONLY | CL_MEM_WRITE_ONLY | CL_MEM_READ_WRITE;



device::OCLBuffer::OCLBuffer(cl_mem buffer, OCLDevice dev) noexcept
    : m_device(std::move(dev)),
      m_buffer(buffer)
{}


BufferMode OCLBuffer::mode() const
{
    RPY_DBG_ASSERT(m_buffer != nullptr);
    cl_int mode;
    auto ecode = clGetMemObjectInfo(
            m_buffer,
            CL_MEM_FLAGS,
            sizeof(mode),
            &mode,
            nullptr
    );
    if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }

    switch (mode & MODE_MASK) {
        case CL_MEM_READ_WRITE: return BufferMode::ReadWrite;
        case CL_MEM_WRITE_ONLY: return BufferMode::Write;
        case CL_MEM_READ_ONLY: return BufferMode::Read;
        default: RPY_THROW(std::runtime_error, "invalid buffer mode");
    }
}
dimn_t OCLBuffer::size() const
{
    RPY_DBG_ASSERT(m_buffer != nullptr);

    cl_ulong size;
    auto ecode = clGetMemObjectInfo(
            m_buffer,
            CL_MEM_SIZE,
            sizeof(cl_ulong),
            &size,
            nullptr
    );
    if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }

    return static_cast<dimn_t>(size);
}
void* OCLBuffer::ptr() { return m_buffer; }
std::unique_ptr<device::dtl::InterfaceBase> OCLBuffer::clone() const
{
    RPY_DBG_ASSERT(m_buffer != nullptr);

    dimn_t buf_size = size();

    cl_int ecode;
    cl_mem new_buffer = clCreateBuffer(
            m_device->context(),
            CL_MEM_ALLOC_HOST_PTR,
            buf_size,
            nullptr,
            &ecode
    );
    if (new_buffer == nullptr) { RPY_HANDLE_OCL_ERROR(ecode); }

    cl_event event;
    ecode = clEnqueueCopyBuffer(
            m_device->default_queue(),
            m_buffer,
            new_buffer,
            0,
            0,
            buf_size,
            0,
            nullptr,
            &event
    );
    if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }

    clWaitForEvents(1, &event);

    return std::make_unique<OCLBuffer>(new_buffer, m_device);
}
Device OCLBuffer::device() const noexcept { return m_device; }
OCLBuffer::~OCLBuffer() {
    if (RPY_LIKELY(m_buffer != nullptr)) {
        auto ecode = clReleaseMemObject(m_buffer);
        RPY_DBG_ASSERT(ecode == CL_SUCCESS);
    }
    m_buffer = nullptr;
}
