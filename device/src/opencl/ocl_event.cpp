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
#include "ocl_handle_errors.h"
#include "ocl_headers.h"
#include "ocl_device.h"

using namespace rpy;
using namespace rpy::device;

struct OCLEventInterface::Data {
    cl_event event;
};

#define cast(content) static_cast<Data*>(content)->event

device::OCLEventInterface::OCLEventInterface(OCLDevice dev) noexcept
    : m_device(std::move(dev))
{}

void* device::OCLEventInterface::create_data(cl_event event) noexcept
{
    return new Data{event};
}
cl_event device::OCLEventInterface::take(void* content) noexcept
{
    auto event = cast(content);
    delete static_cast<Data*>(content);
    return event;
}

void OCLEventInterface::wait(void* content) const
{
    cl_event event = cast(content);
    auto ecode = clWaitForEvents(1, &event);
    if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }
}
EventStatus OCLEventInterface::status(void* content) const
{
    cl_int stat;
    auto ecode = clGetEventInfo(
            cast(content),
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
void* OCLEventInterface::clone(void* content) const
{
    auto ecode = clRetainEvent(cast(content));
    if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }
    return create_data(cast(content));
}
void OCLEventInterface::clear(void* content) const
{
    cl_int ecode = clReleaseEvent(take(content));
    RPY_DBG_ASSERT(ecode == CL_SUCCESS);
}

#undef cast
