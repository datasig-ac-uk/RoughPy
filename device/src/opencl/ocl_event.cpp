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

#include "ocl_event.h"
#include "ocl_device.h"
#include "ocl_handle_errors.h"
#include "ocl_headers.h"

using namespace rpy;
using namespace rpy::devices;

OCLEvent::OCLEvent(cl_event event, OCLDevice dev) noexcept
    : m_event(event),
      m_device(std::move(dev))
{}

void OCLEvent::wait()
{
    RPY_DBG_ASSERT(m_event != nullptr);
    auto ecode = clWaitForEvents(1, &m_event);
    if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }
}
EventStatus OCLEvent::status() const
{
    RPY_DBG_ASSERT(m_event != nullptr);

    cl_int stat;
    auto ecode = clGetEventInfo(
            m_event,
            CL_EVENT_COMMAND_EXECUTION_STATUS,
            sizeof(cl_int),
            &stat,
            nullptr
    );

    if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }

    switch (stat) {
        case CL_SUCCESS: return EventStatus::CompletedSuccessfully;
        case CL_RUNNING: return EventStatus::Running;
        case CL_SUBMITTED: return EventStatus::Submitted;
        case CL_QUEUED: return EventStatus::Queued;
        default: return EventStatus::Error;
    }
}
std::unique_ptr<devices::dtl::InterfaceBase> OCLEvent::clone() const
{
    return std::make_unique<OCLEvent>(m_event, m_device);
}
bool OCLEvent::is_user() const noexcept
{
    RPY_DBG_ASSERT(m_event != nullptr);

    cl_command_type type;
    auto ecode = clGetEventInfo(
            m_event,
            CL_EVENT_COMMAND_TYPE,
            sizeof(cl_command_type),
            &type,
            nullptr
    );

    if (RPY_UNLIKELY(ecode != CL_SUCCESS)) { RPY_HANDLE_OCL_ERROR(ecode); }

    return type == CL_COMMAND_USER;
}
void OCLEvent::set_status(EventStatus status)
{
    cl_int exec_status;
    switch (status) {
        case EventStatus::CompletedSuccessfully:
            exec_status = CL_SUCCESS;
            break;
        case EventStatus::Running: exec_status = CL_RUNNING; break;
        case EventStatus::Submitted: exec_status = CL_SUBMITTED; break;
        case EventStatus::Queued: exec_status = CL_QUEUED; break;
        default:
            RPY_THROW(std::runtime_error, "cannot set event satus to none");
    }

    auto ecode = clSetUserEventStatus(m_event, exec_status);
    if (RPY_UNLIKELY(ecode != CL_SUCCESS)) { RPY_HANDLE_OCL_ERROR(ecode); }
}
Device OCLEvent::device() const noexcept { return m_device; }
DeviceType OCLEvent::type() const noexcept { return DeviceType::OpenCL; }
devices::dtl::InterfaceBase::reference_count_type OCLEvent::ref_count() const
        noexcept
{
    cl_uint rc = 0;
    auto ecode = clGetEventInfo(
            m_event,
            CL_EVENT_REFERENCE_COUNT,
            sizeof(rc),
            &rc,
            nullptr
    );
    RPY_DBG_ASSERT(ecode == CL_SUCCESS);
    return static_cast<reference_count_type>(rc);
}
void* OCLEvent::ptr() noexcept { return m_event; }

const void* OCLEvent::ptr() const noexcept { return m_event; }
devices::dtl::InterfaceBase::reference_count_type OCLEvent::inc_ref() noexcept
{
    reference_count_type rc = ref_count();
    if (RPY_LIKELY(m_event != nullptr)) {
        auto ecode = clRetainEvent(m_event);
        RPY_DBG_ASSERT(ecode == CL_SUCCESS);
    }
    return rc;
}
devices::dtl::InterfaceBase::reference_count_type OCLEvent::dec_ref() noexcept
{
    reference_count_type rc = ref_count();
    if (RPY_LIKELY(m_event != nullptr)) {
        RPY_DBG_ASSERT(rc > 0);
        auto ecode = clRetainEvent(m_event);
        RPY_DBG_ASSERT(ecode == CL_SUCCESS);
    }
    return rc;
}
