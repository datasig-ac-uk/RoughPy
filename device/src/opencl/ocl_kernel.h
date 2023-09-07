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

#ifndef ROUGHPY_OCL_KERNEL_H
#define ROUGHPY_OCL_KERNEL_H

#include <roughpy/device/core.h>
#include <roughpy/device/kernel.h>

#include "open_cl_device.h"

namespace rpy {
namespace device {


class OCLKernelInterface : public KernelInterface{

    struct Data {
        cl_kernel kernel;
        boost::intrusive_ptr<OpenCLDevice> device;
    };

public:

    static inline void* create_data(cl_kernel kernel,
                                    boost::intrusive_ptr<OpenCLDevice> device)
    {
        return new Data { kernel, std::move(device) };
    }


private:

    static inline cl_kernel& ker(void* content) noexcept {
        return static_cast<Data*>(content)->kernel;
    }

    static inline boost::intrusive_ptr<OpenCLDevice>& dev(void* content)
            noexcept {
        return static_cast<Data*>(content)->device;
    }

    inline cl_uint nargs(cl_kernel kernel) const {
        cl_uint ret;
        cl_int errcode = ::clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS,
                                   sizeof(cl_uint), &ret, nullptr);
        RPY_CHECK(errcode == CL_SUCCESS);
        return ret;
    }

    inline cl_context context(cl_kernel kernel) const {
        cl_context ctx = nullptr;
        auto errcode = ::clGetKernelInfo(kernel, CL_KERNEL_CONTEXT,
                                 sizeof(ctx), &ctx, nullptr);
        RPY_CHECK(errcode == CL_SUCCESS);
        return ctx;
    }

    inline cl_program program(cl_kernel kernel) const {
        cl_program prog = nullptr;
        auto errcode = ::clGetKernelInfo(kernel, CL_KERNEL_PROGRAM, sizeof(prog),
                                 &prog, nullptr);
        RPY_CHECK(errcode == CL_SUCCESS);
        return prog;
    }

    static inline void handle_launch_error(cl_int errcode) {
        switch (errcode) {
            case CL_SUCCESS:
                return;
            case CL_INVALID_PROGRAM_EXECUTABLE: RPY_FALLTHROUGH;
            case CL_INVALID_COMMAND_QUEUE: RPY_FALLTHROUGH;
            case CL_INVALID_KERNEL: RPY_FALLTHROUGH;
            case CL_INVALID_CONTEXT: RPY_FALLTHROUGH;
            case CL_INVALID_KERNEL_ARGS: RPY_FALLTHROUGH;
            case CL_INVALID_WORK_DIMENSION: RPY_FALLTHROUGH;
            case CL_INVALID_GLOBAL_WORK_SIZE: RPY_FALLTHROUGH;
            case CL_INVALID_GLOBAL_OFFSET: RPY_FALLTHROUGH;
            case CL_INVALID_WORK_GROUP_SIZE: RPY_FALLTHROUGH;
            case CL_INVALID_WORK_ITEM_SIZE: RPY_FALLTHROUGH;
            case CL_MISALIGNED_SUB_BUFFER_OFFSET: RPY_FALLTHROUGH;
            case CL_DEVICE_MEM_BASE_ADDR_ALIGN: RPY_FALLTHROUGH;
            case CL_INVALID_IMAGE_SIZE: RPY_FALLTHROUGH;
            case CL_DEVICE_MAX_READ_IMAGE_ARGS: RPY_FALLTHROUGH;
            case CL_MEM_OBJECT_ALLOCATION_FAILURE: RPY_FALLTHROUGH;
            case CL_INVALID_EVENT_WAIT_LIST: RPY_FALLTHROUGH;
            case CL_INVALID_OPERATION: RPY_FALLTHROUGH;
            case CL_OUT_OF_RESOURCES: RPY_FALLTHROUGH;
            case CL_OUT_OF_HOST_MEMORY: RPY_FALLTHROUGH;
            default:
                RPY_THROW(std::runtime_error, "kernel launch unsuccessful");
        }
        RPY_UNREACHABLE_RETURN((void) 0);
    }

public:

    void* clone(void* content) const override;
    void clear(void* content) const override;

    string_view name(void* content) const override;
    dimn_t num_args(void* content) const override;

    Event launch_kernel_async(
            void* content,
            Queue queue,
            Slice<void*> args,
            Slice<dimn_t> arg_sizes,
            const KernelLaunchParams& params
            ) const override;

};

namespace cl {
RPY_NO_DISCARD const OCLKernelInterface* kernel_interface() noexcept;
}

}// namespace device
}// namespace rpy

#endif// ROUGHPY_OCL_KERNEL_H
