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


#include "ocl_helpers.h"


using namespace rpy;
using namespace rpy::devices;





cl_mem cl::to_ocl_buffer(void* data, dimn_t size, cl_context context) {
    cl_int ecode;
    auto buffer = clCreateBuffer(context,
                                 CL_MEM_USE_HOST_PTR,
                                 size,
                                 data,
                                 &ecode
    );

    if (buffer == nullptr) {
        RPY_HANDLE_OCL_ERROR(ecode);
    }

    return buffer;
}
Slice<byte> cl::from_ocl_buffer(cl_mem buf) {
    size_t count = 0;
    byte* data = nullptr;
    auto ecode = clGetMemObjectInfo(buf, CL_MEM_SIZE, sizeof(count), &count,
                                    nullptr);
    if (ecode != CL_SUCCESS) {
        RPY_HANDLE_OCL_ERROR(ecode);
    }

    if (count > 0) {
        ecode = clGetMemObjectInfo(buf, CL_MEM_HOST_PTR, sizeof(data), &data,
                                   nullptr);
        if (ecode != CL_SUCCESS) {
            RPY_HANDLE_OCL_ERROR(ecode);
        }
    }

    return {data, count};
}
