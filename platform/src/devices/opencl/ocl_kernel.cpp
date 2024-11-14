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

#include "roughpy/core/check.h"               // for throw_exception, RPY_THROW
#include "roughpy/core/debug_assertion.h"     // for RPY_DBG_ASSERT

#include "ocl_device.h"
#include "ocl_handle_errors.h"
#include "ocl_helpers.h"

#include "devices/queue.h"

using namespace rpy;
using namespace rpy::devices;

OCLKernel::OCLKernel(cl_kernel kernel, OCLDevice dev) noexcept
    : m_kernel(kernel),
      m_device(std::move(dev))
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

namespace {

void check_and_set_argument(
        cl_kernel kernel,
        OCLDevice device,
        cl_uint arg_idx,
        const KernelArgument& arg
)
{
    cl_int ecode = CL_SUCCESS;
    if (arg.is_buffer()) {
        /*
         * Buffers need special attention here because they might not be
         * buffers allocated using OpenCL or on the same device. Currently,
         * we're only going to use OpenCL buffers that have the correct device.
         *
         * We might handle this differently in the future.
         */
        const auto& buf_ref = arg.const_buffer();
        if (buf_ref.device() == device) {
            ecode = clSetKernelArg(
                    kernel,
                    arg_idx,
                    sizeof(cl_mem),
                    buf_ref.ptr()
            );
        } else {
            RPY_THROW(
                    std::invalid_argument,
                    "kernel argument " + std::to_string(arg_idx)
                            + " is allocated to a device which cannot be used "
                              "with the OpenCL backend"
            );
        }
    } else {
        /*
         * If the argument is a simple pointer then we can just set it as usual
         * without any need for complicated type checking.
         */
        ecode = clSetKernelArg(
                kernel,
                arg_idx,
                arg.size(),
                arg.const_pointer()
        );
    }

    /*
     * To provide better error feedback to the user, we customise the error
     * messages for some of the errors rather than deferring to
     * RPY_HANDLE_OCL_ERROR.
     */
    switch (ecode) {
        case CL_SUCCESS: break;
        case CL_INVALID_ARG_VALUE:
        case CL_INVALID_ARG_SIZE:
            RPY_THROW(
                    std::invalid_argument,
                    "the argument at index " + std::to_string(arg_idx)
                            + " has an invalid value"
            );
        case CL_MAX_SIZE_RESTRICTION_EXCEEDED:
            RPY_THROW(
                    std::invalid_argument,
                    "the size of argument " + std::to_string(arg_idx)
                            + " exceeds the maximum admissible size"
            );
        default: RPY_HANDLE_OCL_ERROR(ecode);
    }
}

}// namespace

Event OCLKernel::launch_kernel_async(
        Queue& queue,
        const KernelLaunchParams& params,
        Slice<KernelArgument> args
)
{
    RPY_DBG_ASSERT(m_kernel != nullptr);

    auto n_args = num_args();
    RPY_DBG_ASSERT(args.size() == n_args);

    cl_int ecode = CL_SUCCESS;

    /*
     * The Kernel class has already checked that queue is either a default queue
     * (i.e. it is empty) or it has the correct device. Thus casting from
     * queue.ptr() to a cl_command_queue is perfectly safe.
     */
    auto queue_to_use = (
            (queue.is_default()) ? m_device->default_queue()
                                 : static_cast<cl_command_queue>(queue.ptr())
    );

    // RAII vector of buffer objects, properly clears on exit

    for (cl_uint i = 0; i < n_args; ++i) {
        check_and_set_argument(m_kernel, m_device, i, args[i]);
    }

    cl_uint work_dim = params.num_dims();
    auto work_size = params.total_work_dims();
    auto group_size = params.work_groups();
    dimn_t gw_size[3] = {work_size.x, work_size.y, work_size.z};
    dimn_t lw_size[3] = {group_size.x, group_size.y, group_size.z};

    cl_event event;
    ecode = clEnqueueNDRangeKernel(
            queue_to_use, /* command_queue */
            m_kernel,     /* kernel */
            work_dim,     /* work_dim */
            nullptr,      /* global_work_offset */
            gw_size,      /* global_work_size */
            lw_size,      /* local_work_size */
            0,            /* num_events_in_wait_list */
            nullptr,      /* event_wait_list */
            &event        /* event */
    );

    if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }

    return m_device->make_event(event);
}
std::unique_ptr<devices::dtl::InterfaceBase> OCLKernel::clone() const
{
    RPY_DBG_ASSERT(m_kernel);
    cl_int ecode = CL_SUCCESS;
    cl_kernel new_ker = clCloneKernel(m_kernel, &ecode);

    if (new_ker == nullptr) { RPY_HANDLE_OCL_ERROR(ecode); }

    return std::make_unique<OCLKernel>(new_ker, m_device);
}
devices::dtl::InterfaceBase::reference_count_type
OCLKernel::ref_count() const noexcept
{
    cl_uint rc = 0;
    auto ecode = clGetKernelInfo(
            m_kernel,
            CL_KERNEL_REFERENCE_COUNT,
            sizeof(rc),
            &rc,
            nullptr
    );
    if (ecode != CL_SUCCESS) { return 0; }
    return static_cast<reference_count_type>(rc);
}
Device OCLKernel::device() const noexcept { return m_device; }
DeviceType OCLKernel::type() const noexcept { return DeviceType::OpenCL; }
void* OCLKernel::ptr() noexcept { return m_kernel; }
const void* OCLKernel::ptr() const noexcept { return m_kernel; }
devices::dtl::InterfaceBase::reference_count_type OCLKernel::inc_ref() noexcept
{
    reference_count_type rc = ref_count();
    if (RPY_LIKELY(m_kernel != nullptr)) {
        auto ecode = clRetainKernel(m_kernel);
        RPY_DBG_ASSERT(ecode == CL_SUCCESS);
    }
    return rc;
}
devices::dtl::InterfaceBase::reference_count_type OCLKernel::dec_ref() noexcept
{
    reference_count_type rc = ref_count();
    if (RPY_LIKELY(m_kernel != nullptr)) {
        RPY_DBG_ASSERT(rc > 0);
        auto ecode = clReleaseKernel(m_kernel);
        RPY_DBG_ASSERT(ecode == CL_SUCCESS);
    }
    return rc;
}
