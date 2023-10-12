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

#include "ocl_device.h"

#include "ocl_buffer.h"
#include "ocl_event.h"
#include "ocl_handle_errors.h"
#include "ocl_kernel.h"
#include "ocl_queue.h"

using namespace rpy;
using namespace rpy::device;

OCLDeviceHandle::OCLDeviceHandle(cl_device_id id)
    : m_buffer_iface(this),
      m_event_iface(this),
      m_kernel_iface(this),
      m_queue_iface(this),
      m_device(id)
{}

OCLDeviceHandle::~OCLDeviceHandle()
{
    clReleaseCommandQueue(m_default_queue);

    m_device_id = 0;
    m_default_queue = nullptr;

    cl_int ecode;
    while (!m_programs.empty()) {
        ecode = clReleaseProgram(m_programs.back());
        RPY_DBG_ASSERT(ecode == CL_SUCCESS);
        m_programs.pop_back();
    }

    for (auto&& [nm, ker] : m_kernels) {
        ecode = clReleaseKernel(ker);
        RPY_DBG_ASSERT(ecode == CL_SUCCESS);
    }
    m_kernels.clear();

    clReleaseContext(m_ctx);
    clReleaseDevice(m_device);
}
DeviceInfo OCLDeviceHandle::info() const noexcept
{
    return {DeviceType::OpenCL, m_device_id};
}
optional<fs::path> OCLDeviceHandle::runtime_library() const noexcept
{
    return {};
}
Buffer OCLDeviceHandle::raw_alloc(dimn_t count, dimn_t alignment) const
{
    cl_int ecode;
    auto new_mem
            = clCreateBuffer(m_ctx, CL_MEM_READ_WRITE, count, nullptr, &ecode);

    if (new_mem == nullptr) { RPY_HANDLE_OCL_ERROR(ecode); }
    return Buffer(buffer_interface(), OCLBufferInterface::create_data(new_mem));
}
void OCLDeviceHandle::raw_free(Buffer buffer) const
{
    RPY_CHECK(buffer.interface() == buffer_interface());
    auto buf = OCLBufferInterface::take(buffer.content());
    auto ecode = clReleaseMemObject(buf);
    RPY_DBG_ASSERT(ecode == CL_SUCCESS);
}
const BufferInterface* OCLDeviceHandle::buffer_interface() const noexcept
{
    return &m_buffer_iface;
}
const EventInterface* OCLDeviceHandle::event_interface() const noexcept
{
    return &m_event_iface;
}
const KernelInterface* OCLDeviceHandle::kernel_interface() const noexcept
{
    return &m_kernel_iface;
}
const QueueInterface* OCLDeviceHandle::queue_interface() const noexcept
{
    return &m_queue_iface;
}
