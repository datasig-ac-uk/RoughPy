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
// Created by sam on 25/08/23.
//

#include "open_cl_kernel.h"


using namespace rpy;
using namespace device;

OCLKernelInterface::OCLKernelInterface(const OpenCLRuntimeLibrary* runtime) {}
void* OCLKernelInterface::clone(void* content) const
{
    cl_int ecode = CL_SUCCESS;
    cl_kernel new_kernel = clone_impl(ker(content),
                                      &ecode);
    RPY_CHECK(ecode == CL_SUCCESS);
    return new Data { new_kernel, dev(content) };
}
void OCLKernelInterface::clear(void* content) const
{
    auto* as_data = static_cast<Data*>(content);
    release_impl(as_data->kernel);
    delete as_data;
}
string_view OCLKernelInterface::name(void* content) const
{
    char* k_name;
    auto errcode = info_impl(ker(content),
                             CL_KERNEL_CONTEXT,
                             sizeof(k_name), &k_name, nullptr);
    RPY_CHECK(errcode == CL_SUCCESS);
    return {k_name};
}
dimn_t OCLKernelInterface::num_args(void* content) const
{
    return nargs(ker(content));
}

Event OCLKernelInterface::launch_kernel_async(
        void* content, Queue queue, Slice<void*> args, Slice<dimn_t> arg_sizes,
        const KernelLaunchParams& params
) const
{
    auto& kernel = ker(content);
    auto& device = dev(content);
    RPY_DBG_ASSERT(args.size() == arg_sizes.size());

    auto n_args = nargs(kernel);
    RPY_DBG_ASSERT(args.size() == n_args);

    for (dimn_t i = 0; i < n_args; ++i) {
#ifdef RPY_DEBUG

#endif
        set_arg_impl(kernel, i, arg_sizes[i], args[i]);
    }

    cl_uint work_dim = 1;
    dimn_t gw_size[3];
    dimn_t lw_size[3];

    // This is messy, but it gets the job done.
    gw_size[0] = params.grid_work_size.x;
    lw_size[0] = params.grid_work_size.x;
    if (params.grid_work_size.y > 1) {
        gw_size[1] = params.grid_work_size.y;
        lw_size[1] = params.block_work_size.y;
        ++work_dim;

        if (params.grid_work_size.z > 1) {
            gw_size[2] = params.grid_work_size.z;
            lw_size[2] = params.block_work_size.z;
            ++work_dim;
        }
    }

    cl_command_queue command_queue;
    if (queue.interface() != nullptr && queue.content() != nullptr) {
        RPY_CHECK(queue.interface() == p_runtime->queue_interface());
        command_queue = static_cast<cl_command_queue>(queue.content());
    } else {
        command_queue = device->cl_default_queue();
    }

    cl_event event;
    cl_int ecode = enqueue_impl(
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

    handle_launch_error(ecode);


}
