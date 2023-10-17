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
// Created by user on 16/10/23.
//

#ifndef ROUGHPY_DEVICE_SRC_OPENCL_OCL_HELPERS_H_
#define ROUGHPY_DEVICE_SRC_OPENCL_OCL_HELPERS_H_

#include "ocl_handle_errors.h"
#include "ocl_headers.h"

#include <roughpy/core/slice.h>
#include <roughpy/core/types.h>

namespace rpy {
namespace device {
namespace cl {

template <typename Fn, typename CLObj, typename Info>
RPY_NO_DISCARD inline string
string_info(Fn&& fn, CLObj* cl_object, Info info_id)
{
    size_t ret_size;
    auto ecode = fn(cl_object, info_id, 0, nullptr, &ret_size);
    if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }

    string result;
    result.resize(ret_size);
    ecode = fn(cl_object, info_id, result.size(), result.data(), nullptr);
    if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }

    return result;
}

RPY_NO_DISCARD
cl_mem to_ocl_buffer(void* data, dimn_t size, cl_context context);

RPY_NO_DISCARD
Slice<byte> from_ocl_buffer(cl_mem buf);

}// namespace cl
}// namespace device
}// namespace rpy

#endif// ROUGHPY_DEVICE_SRC_OPENCL_OCL_HELPERS_H_
