// Copyright (c) 2023 Datasig Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
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
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

//
// Created by user on 28/04/23.
//

#ifndef ROUGHPY_DEVICE_KERNELS_UNARY_KERNEL_CUH
#define ROUGHPY_DEVICE_KERNELS_UNARY_KERNEL_CUH

#include "core.h"
#include "functors.h"
#include <roughpy/core/implementation_types.h>
#include <roughpy/core/macros.h>

namespace rpy {
namespace device {

template <typename S, typename Functor>
RPY_KERNEL void unary_kernel(
        S* RPY_RESTRICT dst, const S* RPY_RESTRICT src, dindex_t count,
        Functor&& fn
)
{
    dindex_t offset = blockIdx.x * gridDim.x + threadIdx.x;
    dindex_t stride = blockDim.x * gridDim.x;

    for (dindex_t i = offset; i < count; i += stride) { dst[i] = fn(src[i]); }
}

}// namespace device
}// namespace rpy

#endif// ROUGHPY_DEVICE_KERNELS_UNARY_KERNEL_CUH
