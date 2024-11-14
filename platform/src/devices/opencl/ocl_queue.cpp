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


#include "roughpy/core/debug_assertion.h"  // for RPY_DBG_ASSERT

#include "ocl_device.h"
#include "ocl_handle_errors.h"
#include "ocl_headers.h"

using namespace rpy;
using namespace rpy::devices;

OCLQueue::OCLQueue(cl_command_queue queue, OCLDevice dev) noexcept
    : m_queue(queue),
      m_device(std::move(dev))
{}

std::unique_ptr<rpy::devices::dtl::InterfaceBase> OCLQueue::clone() const
{
    RPY_DBG_ASSERT(m_queue != nullptr);
    auto ecode = clRetainCommandQueue(m_queue);
    if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }
    return std::make_unique<OCLQueue>(m_queue, m_device);
}

dimn_t OCLQueue::size() const
{
    RPY_DBG_ASSERT(m_queue != nullptr);
    cl_int ecode;
    cl_uint sz;
    ecode = clGetCommandQueueInfo(
            m_queue,
            CL_QUEUE_SIZE,
            sizeof(cl_uint),
            &sz,
            nullptr
    );

    if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }

    return static_cast<dimn_t>(sz);
}
Device OCLQueue::device() const noexcept { return m_device; }
OCLQueue::~OCLQueue()
{
    m_queue = nullptr;
}
DeviceType OCLQueue::type() const noexcept { return DeviceType::OpenCL; }

devices::dtl::InterfaceBase::reference_count_type
OCLQueue::ref_count() const noexcept
{
    cl_uint rc = 0;
    auto ecode = clGetCommandQueueInfo(
            m_queue,
            CL_QUEUE_REFERENCE_COUNT,
            sizeof(rc),
            &rc,
            nullptr
    );
    RPY_DBG_ASSERT(ecode == CL_SUCCESS);
    return rc;
}
void* OCLQueue::ptr() noexcept { return m_queue; }
const void* OCLQueue::ptr() const noexcept { return m_queue; }
devices::dtl::InterfaceBase::reference_count_type OCLQueue::inc_ref() noexcept
{
    reference_count_type rc = ref_count();
    if (RPY_LIKELY(m_queue != nullptr)) {
        auto ecode = clRetainCommandQueue(m_queue);
        RPY_DBG_ASSERT(ecode == CL_SUCCESS);
    }
    return rc;
}
devices::dtl::InterfaceBase::reference_count_type OCLQueue::dec_ref() noexcept
{
    reference_count_type rc = ref_count();
    if (RPY_LIKELY(m_queue != nullptr)) {
        RPY_DBG_ASSERT(rc > 0);
        auto ecode = clReleaseCommandQueue(m_queue);
        RPY_DBG_ASSERT(ecode == CL_SUCCESS);
    }
    return rc;
}
