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

#include "ocl_queue.h"

#include "ocl_handle_errors.h"
#include "ocl_headers.h"
#include "ocl_device.h"

using namespace rpy;
using namespace rpy::device;

struct OCLQueueInterface::Data {
    cl_command_queue queue;
};

#define cast(content) static_cast<Data*>(content)->queue

device::OCLQueueInterface::OCLQueueInterface(OCLDevice dev) noexcept
    : m_device(std::move(dev))
{

}

void* device::OCLQueueInterface::create_data(cl_command_queue queue) noexcept
{
    return new Data { queue };
}
cl_command_queue device::OCLQueueInterface::take(void* content) noexcept
{
    auto queue = cast(content);
    delete static_cast<Data*>(content);
    return queue;
}

void* OCLQueueInterface::clone(void* content) const
{
    auto ecode = clRetainCommandQueue(cast(content));
    if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }
    return create_data(cast(content));
}
void OCLQueueInterface::clear(void* content) const
{
    auto ecode = clReleaseCommandQueue(take(content));
    if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }
}

dimn_t OCLQueueInterface::size(void* content) const
{
    cl_int ecode;
    cl_ulong sz;
    ecode = clGetCommandQueueInfo(
            cast(content),
            CL_QUEUE_SIZE,
            sizeof(cl_ulong),
            &sz,
            nullptr
    );

    if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }

    return static_cast<dimn_t>(sz);
}
