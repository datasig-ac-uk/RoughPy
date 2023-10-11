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

#ifndef ROUGHPY_DEVICE_SRC_OPENCL_OCL_KERNEL_H_
#define ROUGHPY_DEVICE_SRC_OPENCL_OCL_KERNEL_H_

#include <roughpy/device/kernel.h>

#include "ocl_headers.h"
#include "ocl_device.h"

namespace rpy {
namespace device {
namespace cl {

class OCLKernelInterface : public KernelInterface
{
    struct Data {
        cl_kernel kernel;
        OCLDevice device;
    };

    static inline cl_kernel& ker(void* content) noexcept {
        return static_cast<Data*>(content)->kernel;
    }

    static inline const OCLDevice& dev(void* content) noexcept {
        return static_cast<Data*>(content)->device;
    }


    RPY_UNUSED
    cl_program program(cl_kernel kernel) const;
    RPY_UNUSED
    cl_context context(cl_kernel kernel) const;


public:

    static inline void* create_data(cl_kernel k,
                                    OCLDevice d) {
        return new Data{k, std::move(d)};
    }

    void* clone(void* content) const override;
    void clear(void* content) const override;

    string_view name(void* content) const override;
    dimn_t num_args(void* content) const override;
    Event launch_kernel_async(
            void* content,
            Queue& queue,
            Slice<void*> args,
            Slice<dimn_t> arg_sizes,
            const KernelLaunchParams& params
    ) const override;
};

}// namespace cl
}// namespace device
}// namespace rpy

#endif// ROUGHPY_DEVICE_SRC_OPENCL_OCL_KERNEL_H_
