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

#include "ocl_event.h"

namespace rpy {
namespace device {
void* OCLEventInterface::clone(void* content) const
{
    auto errcode = p_runtime->clRetainEvent(event(content));
    RPY_CHECK(errcode == CL_SUCCESS);
    return content;
}
void OCLEventInterface::clear(void* content) const
{
    auto ecode = p_runtime->clReleaseEvent(event(content));
    RPY_CHECK(ecode == CL_SUCCESS);
}
void OCLEventInterface::wait(void* content) {
    auto ev = event(content);
    auto ecode = p_runtime->clWaitForEvents(1, &ev);
    RPY_CHECK(ecode == CL_SUCCESS);
}
EventStatus OCLEventInterface::status(void* content)
{
    cl_int raw_status;
    auto ecode = p_runtime->clGetEventInfo(event(content),
                              CL_EVENT_COMMAND_EXECUTION_STATUS,
                              sizeof(raw_status),
                              &raw_status,
                              nullptr
                              );
    RPY_CHECK(ecode == CL_SUCCESS);

    switch (raw_status) {
        case CL_COMPLETE:
            return EventStatus::CompletedSuccessfully;
        case CL_QUEUED:
            return EventStatus::Queued;
        case CL_SUBMITTED:
            return EventStatus::Submitted;
        case CL_RUNNING:
            return EventStatus::Running;
        default:
            return EventStatus::Error;
    }

}
}// namespace device
}// namespace rpy
