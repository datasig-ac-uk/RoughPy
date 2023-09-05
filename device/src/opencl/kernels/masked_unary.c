

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

#include "kernel_types.h"


RPY_KERNEL void masked_uminus_float(
        RPY_ADDR_GLOBL float* dst,
        RPY_ADDR_GLOBL const float* src,
        RPY_ADDR_GLOBL const mask_kt* mask
        )
{
    size_t global_idx = get_global_id(0);
    size_t local_idx = get_local_id(0);

    if (mask) {
        const size_t mask_idx = global_idx / RPY_MASK_ELT_SIZE;
        const size_t local_size = get_local_size(0);

        for (; local_idx < 8*sizeof(mask_kt); local_idx += local_size) {
            if (mask[mask_idx] & (1 << local_idx)) {
                dst[global_idx] = -src[global_idx];
            }
        }
    } else {
        dst[global_idx] = -src[global_idx];
    }
}

RPY_KERNEL void masked_uminus_double(
        RPY_ADDR_GLOBL double* dst,
        RPY_ADDR_GLOBL const double* src,
        RPY_ADDR_GLOBL const mask_kt* mask
        )
{
    size_t global_idx = get_global_id(0);

    if (mask) {
        size_t local_idx = get_local_id(0);
        const size_t mask_idx = global_idx / RPY_MASK_ELT_SIZE;
        const size_t local_size = get_local_size(0);

        for (; local_idx < 8*sizeof(mask_kt); local_idx += local_size) {
            if (mask[mask_idx] & (1 << local_idx)) {
                dst[global_idx] = -src[global_idx];
            }
        }
    } else {
        dst[global_idx] = -src[global_idx];
    }
}


RPY_KERNEL void masked_uminus_half(
        RPY_ADDR_GLOBL half* dst,
        RPY_ADDR_GLOBL const half* src,
        RPY_ADDR_GLOBL const mask_kt* mask
        )
{
    size_t global_idx = get_global_id(0);
    size_t local_idx = get_local_id(0);

    if (mask) {
        const size_t mask_idx = global_idx / RPY_MASK_ELT_SIZE;
        const size_t local_size = get_local_size(0);

        for (; local_idx < 8*sizeof(mask_kt); local_idx += local_size) {
            if (mask[mask_idx] & (1 << local_idx)) {
                dst[global_idx] = -src[global_idx];
            }
        }
    } else {
        dst[global_idx] = -src[global_idx];
    }
}
