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

#include "ocl_kernel.h"

#include "ocl_device.h"
#include "ocl_handle_errors.h"
#include "ocl_helpers.h"

#include <roughpy/device/queue.h>

using namespace rpy;
using namespace rpy::device;

OCLKernel::OCLKernel(cl_kernel kernel, OCLDevice dev) noexcept
    : m_kernel(kernel), m_device(std::move(dev))
{}



cl_program OCLKernel::program() const
{
    RPY_DBG_ASSERT(m_kernel != nullptr);
    cl_program prog = nullptr;
    auto ecode = clGetKernelInfo(
            m_kernel,
            CL_KERNEL_PROGRAM,
            sizeof(cl_program),
            &prog,
            nullptr
    );

    if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }

    RPY_DBG_ASSERT(prog);
    return prog;
}
cl_context OCLKernel::context() const
{
    RPY_DBG_ASSERT(m_kernel != nullptr);
    cl_context ctx = nullptr;
    auto ecode = clGetKernelInfo(
            m_kernel,
            CL_KERNEL_CONTEXT,
            sizeof(cl_context),
            &ctx,
            nullptr
    );

    if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }

    RPY_DBG_ASSERT(ctx);

    return ctx;
}

string OCLKernel::name() const
{
    return cl::string_info(clGetKernelInfo, m_kernel, CL_KERNEL_FUNCTION_NAME);
}
dimn_t OCLKernel::num_args() const
{
    RPY_DBG_ASSERT(m_kernel != nullptr);
    cl_int ecode = CL_SUCCESS;
    cl_uint nargs = 0;
    ecode = clGetKernelInfo(
            m_kernel,
            CL_KERNEL_NUM_ARGS,
            sizeof(cl_uint),
            &nargs,
            nullptr
    );
    if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }

    return static_cast<dimn_t>(nargs);
}
Event OCLKernel::launch_kernel_async(
        Queue& queue,
        Slice<void*> args,
        Slice<dimn_t> arg_sizes,
        const KernelLaunchParams& params
)
{
    RPY_DBG_ASSERT(m_kernel != nullptr);
    RPY_DBG_ASSERT(args.size() == arg_sizes.size());

    auto n_args = num_args();
    RPY_DBG_ASSERT(args.size() == n_args);

    cl_int ecode = CL_SUCCESS;

    for (dimn_t i = 0; i < n_args; ++i) {
        ecode = clSetKernelArg(m_kernel, i, arg_sizes[i], args[i]);
        if (ecode != CL_SUCCESS) {
            RPY_HANDLE_OCL_ERROR(ecode);
        }
    }

    cl_uint work_dim = 1;
    dimn_t gw_size[3];
    dimn_t lw_size[3];

    // This is messy, but it gets the job done.
    //    gw_size[0] = params.grid_work_size.x;
    //    lw_size[0] = params.grid_work_size.x;
    //    if (params.grid_work_size.y > 1) {
    //        gw_size[1] = params.grid_work_size.y;
    //        lw_size[1] = params.block_work_size.y;
    //        ++work_dim;
    //
    //        if (params.grid_work_size.z > 1) {
    //            gw_size[2] = params.grid_work_size.z;
    //            lw_size[2] = params.block_work_size.z;
    //            ++work_dim;
    //        }
    //    }

    cl_command_queue command_queue = m_device->default_queue();
    // TODO: Use queue.is_default() to test whether to fill in the default
//    if (queue.interface() != nullptr && queue.content() != nullptr) {
//        RPY_CHECK(queue.interface() == m_device->queue_interface());
//        command_queue = static_cast<cl_command_queue>(queue.content());
//    } else {
//        command_queue = m_device->default_queue();
//    }

    cl_event event;
    ecode = clEnqueueNDRangeKernel(
            command_queue, /* kernel */
            m_kernel,        /* kernel */
            work_dim,      /* work_dim */
            nullptr,       /* global_work_offset */
            gw_size,       /* global_work_size */
            lw_size,       /* local_work_size */
            0,             /* num_events_in_wait_list */
            nullptr,       /* event_wait_list */
            &event         /* event */
    );

    if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }

    return Event(std::make_unique<OCLEvent>(event, m_device));
}
std::unique_ptr<device::dtl::InterfaceBase> OCLKernel::clone() const
{
    RPY_DBG_ASSERT(m_kernel);
    cl_int ecode= CL_SUCCESS;
    cl_kernel new_ker = clCloneKernel(m_kernel, &ecode);

    if (new_ker == nullptr) { RPY_HANDLE_OCL_ERROR(ecode); }

    return std::make_unique<OCLKernel>(new_ker, m_device);
}
