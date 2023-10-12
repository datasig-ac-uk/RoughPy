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

#include "ocl_handle_errors.h"
#include "ocl_device.h"

#include <roughpy/device/queue.h>

using namespace rpy;
using namespace rpy::device;

struct OCLKernelInterface::Data {
    cl_kernel kernel;
};

#define ker(content) static_cast<Data*>(content)->kernel
#define dev(content) m_device

device::OCLKernelInterface::OCLKernelInterface(OCLDevice dev) noexcept
    : m_device(std::move(dev))
{}

void* device::OCLKernelInterface::create_data(cl_kernel k) noexcept
{
    return new Data { k };
}
cl_kernel device::OCLKernelInterface::take(void* content) noexcept
{
    auto k = ker(content);
    ker(content) = nullptr;
    delete static_cast<Data*>(content);
    return k;
}

cl_program OCLKernelInterface::program(cl_kernel kernel) const
{
    cl_program prog;
    auto ecode = clGetKernelInfo(
            kernel,
            CL_KERNEL_PROGRAM,
            sizeof(cl_program),
            &prog,
            nullptr
    );

    if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }

    RPY_DBG_ASSERT(prog);
    return prog;
}
cl_context OCLKernelInterface::context(cl_kernel kernel) const
{
    cl_context ctx;
    auto ecode = clGetKernelInfo(
            kernel,
            CL_KERNEL_CONTEXT,
            sizeof(cl_context),
            &ctx,
            nullptr
    );

    if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }

    RPY_DBG_ASSERT(ctx);

    return ctx;
}

string_view OCLKernelInterface::name(void* content) const
{
    cl_int ecode;
    char* cl_name;
    cl_ulong cl_name_len;
    ecode = clGetKernelInfo(
            ker(content),
            CL_KERNEL_FUNCTION_NAME,
            sizeof(char*),
            &cl_name,
            &cl_name_len
    );
    if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }

    return {cl_name, cl_name_len};
}
dimn_t OCLKernelInterface::num_args(void* content) const
{
    cl_int ecode;
    cl_uint nargs;
    ecode = clGetKernelInfo(
            ker(content),
            CL_KERNEL_NUM_ARGS,
            sizeof(cl_uint),
            &nargs,
            nullptr
    );
    if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }

    return static_cast<dimn_t>(nargs);
}
Event OCLKernelInterface::launch_kernel_async(
        void* content,
        Queue& queue,
        Slice<void*> args,
        Slice<dimn_t> arg_sizes,
        const KernelLaunchParams& params
) const
{

    auto& kernel = ker(content);

    RPY_DBG_ASSERT(args.size() == arg_sizes.size());

    auto n_args = num_args(kernel);
    RPY_DBG_ASSERT(args.size() == n_args);

    for (dimn_t i = 0; i < n_args; ++i) {
#ifdef RPY_DEBUG

#endif
        clSetKernelArg(kernel, i, arg_sizes[i], args[i]);
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

    cl_command_queue command_queue;
    if (queue.interface() != nullptr && queue.content() != nullptr) {
        RPY_CHECK(queue.interface() == m_device->queue_interface());
        command_queue = static_cast<cl_command_queue>(queue.content());
    } else {
        command_queue = m_device->default_queue();
    }

    cl_event event;
    cl_int ecode = clEnqueueNDRangeKernel(
            command_queue, /* kernel */
            kernel,        /* kernel */
            work_dim,      /* work_dim */
            nullptr,       /* global_work_offset */
            gw_size,       /* global_work_size */
            lw_size,       /* local_work_size */
            0,             /* num_events_in_wait_list */
            nullptr,       /* event_wait_list */
            &event         /* event */
    );

    if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }

    return Event{m_device->event_interface(), event};
}
void* OCLKernelInterface::clone(void* content) const
{
    cl_int ecode;
    cl_kernel new_ker = clCloneKernel(ker(content), &ecode);

    if (new_ker == nullptr) { RPY_HANDLE_OCL_ERROR(ecode); }

    return create_data(new_ker);
}
void OCLKernelInterface::clear(void* content) const
{
    auto ecode = clReleaseKernel(take(content));
    RPY_DBG_ASSERT(ecode == CL_SUCCESS);
}
