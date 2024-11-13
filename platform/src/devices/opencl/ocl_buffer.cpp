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

#include <roughpy/core/check.h>
#include <roughpy/core/debug_assertion.h>

#include "ocl_device.h"
#include "ocl_handle_errors.h"
#include "ocl_headers.h"
#include "ocl_helpers.h"

#include "devices/host_device.h"

#include <utility>

using namespace rpy;
using namespace rpy::devices;

constexpr cl_int MODE_MASK
        = CL_MEM_READ_ONLY | CL_MEM_WRITE_ONLY | CL_MEM_READ_WRITE;

devices::OCLBuffer::OCLBuffer(cl_mem buffer, OCLDevice dev) noexcept
    : m_device(std::move(dev)),
      m_buffer(buffer)
{}
OCLBuffer::~OCLBuffer() { m_buffer = nullptr; }

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
void* OCLBuffer::ptr() noexcept { return &m_buffer; }
std::unique_ptr<devices::dtl::InterfaceBase> OCLBuffer::clone() const
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

devices::dtl::InterfaceBase::reference_count_type
OCLBuffer::ref_count() const noexcept
{
    if (m_buffer != nullptr) {
        cl_uint ref_count = 0;
        auto ecode = clGetMemObjectInfo(
                m_buffer,
                CL_MEM_REFERENCE_COUNT,
                sizeof(ref_count),
                &ref_count,
                nullptr
        );
        RPY_DBG_ASSERT(ecode == CL_SUCCESS);
        return static_cast<dimn_t>(ref_count);
    }
    return 0;
}
DeviceType OCLBuffer::type() const noexcept { return DeviceType::OpenCL; }
const void* OCLBuffer::ptr() const noexcept { return &m_buffer; }
Event OCLBuffer::to_device(Buffer& dst, const Device& device, Queue& queue)
{
    if (device == m_device) {
        dst = clone_cast(this);
        return Event();
    }

    if (device == get_host_device()) {
        return m_device->to_host(dst, *this, queue);
    }

    auto queue_to_use_this = cl::scoped_guard(
            queue.is_default() ? m_device->default_queue()
                               : static_cast<cl_command_queue>(queue.ptr())
    );

    auto buffer_size = size();

    if (device->type() == DeviceType::OpenCL) {
        // The device is an OpenCL device but is not this device.
        // We can safely cast up to a OCLDeviceHandle from the raw DeviceHandle.
        const OCLDeviceHandle& ocl_handle
                = static_cast<const OCLDeviceHandle&>(*device);

        /*
         * The procedure is going to be as follows:
         *  1) Map the source buffer (this) into host-accessible memory.
         *  2) Write the contents of the mapped buffer into the destination
         *  buffer (dst).
         *
         *  There might be a better way to do this if both platforms share
         *  the same platform or are otherwise related.
         */
        cl_int ecode = CL_SUCCESS;
        auto map_event = cl::scoped_guard(static_cast<cl_event>(nullptr));
        auto* mapped_data = clEnqueueMapBuffer(
                queue_to_use_this,
                m_buffer,
                CL_FALSE,
                CL_MAP_READ,
                0,
                buffer_size,
                0,
                nullptr,
                map_event.ptr(),
                &ecode
        );

        if (mapped_data == nullptr) { RPY_HANDLE_OCL_ERROR(ecode); }

        cl_uint wait_events = 0;
        auto write_event = cl::scoped_guard(static_cast<cl_event>(nullptr));
        if (m_device->context() == ocl_handle.context()) {
            /*
             * Things are dramatically simpler if the devices have the same
             * context since we can just reuse some of the parts.
             */
            ecode = clEnqueueWriteBuffer(
                    queue_to_use_this,
                    *static_cast<cl_mem*>(dst.ptr()),
                    CL_FALSE,
                    0,
                    buffer_size,
                    mapped_data,
                    1,
                    map_event.ptr(),
                    write_event.ptr()
            );

            if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }
            wait_events = 1;
        } else {
            /*
             * I don't know whether events can be passed between contexts
             * safely, so let's assume not. So now we have to wait on the map
             * event, then perform a blocking write to dst.
             */
            ecode = clWaitForEvents(1, map_event.ptr());

            if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }

            auto queue_to_use_other
                    = cl::scoped_guard(ocl_handle.default_queue());

            ecode = clEnqueueWriteBuffer(
                    queue_to_use_other,
                    *static_cast<cl_mem*>(dst.ptr()),
                    CL_TRUE,
                    0,
                    buffer_size,
                    mapped_data,
                    0,
                    nullptr,
                    nullptr
            );

            if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }
        }

        cl_event unmap_event;
        ecode = clEnqueueUnmapMemObject(
                queue_to_use_this,
                m_buffer,
                mapped_data,
                wait_events,
                (wait_events == 0) ? nullptr : write_event.ptr(),
                &unmap_event
        );

        if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }

        return m_device->make_event(unmap_event);
    }

    return BufferInterface::to_device(dst, device, queue);
}

devices::dtl::InterfaceBase::reference_count_type OCLBuffer::inc_ref() noexcept
{
    reference_count_type rc = ref_count();
    if (m_buffer != nullptr) {
        auto ecode = clRetainMemObject(m_buffer);
        RPY_DBG_ASSERT(ecode == CL_SUCCESS);
    }
    return rc;
}
devices::dtl::InterfaceBase::reference_count_type OCLBuffer::dec_ref() noexcept
{
    reference_count_type rc = ref_count();
    if (RPY_LIKELY(m_buffer != nullptr)) {
        RPY_DBG_ASSERT(rc > 0);
        auto ecode = clReleaseMemObject(m_buffer);
        RPY_DBG_ASSERT(ecode == CL_SUCCESS);
    }
    return rc;
}
void* OCLBuffer::map(BufferMode map_mode, dimn_t size, dimn_t offset)
{
    cl_mem_flags flags;

    auto this_mode = mode();
    if (map_mode == BufferMode::None) { map_mode = this_mode; }

    switch (map_mode) {
        case BufferMode::Read: flags = CL_MAP_READ; break;
        case BufferMode::ReadWrite:
            RPY_CHECK(this_mode != BufferMode::Read);
            flags = CL_MAP_WRITE;
            break;
        case BufferMode::Write:
            RPY_CHECK(this_mode != BufferMode::Read);
            if (m_device->cl_supports_version({1, 2})) {
                flags = CL_MAP_WRITE_INVALIDATE_REGION;
            } else {
                flags = CL_MAP_WRITE;
            }
            break;
        case BufferMode::None: RPY_UNREACHABLE();
    }

    cl_int ecode = CL_SUCCESS;
    void* ptr = clEnqueueMapBuffer(
            m_device->default_queue(),
            m_buffer,
            CL_TRUE,
            flags,
            offset,
            size,
            0,
            nullptr,
            nullptr,
            &ecode
    );

    if (ptr == nullptr) { RPY_HANDLE_OCL_ERROR(ecode); }

    return ptr;
}
void OCLBuffer::unmap(void* ptr) noexcept
{
    if (ptr == nullptr) { return ; }

    auto event = cl::scoped_guard(static_cast<cl_event>(nullptr));
    auto ecode = clEnqueueUnmapMemObject(
            m_device->default_queue(),
            m_buffer,
            ptr,
            0,
            nullptr,
            event.ptr()
    );

    RPY_DBG_ASSERT(ecode == CL_SUCCESS);
    clWaitForEvents(1, event.ptr());
}
