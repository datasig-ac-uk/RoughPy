// copyright (c) 2023 the roughpy developers. all rights reserved.
//
// redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1. redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// this software is provided by the copyright holders and contributors "as is"
// and any express or implied warranties, including, but not limited to, the
// implied warranties of merchantability and fitness for a particular purpose
// are disclaimed. in no event shall the copyright holder or contributors be
// liable for any direct, indirect, incidental, special, exemplary, or consequential
// damages (including, but not limited to, procurement of substitute goods or
// services; loss of use, data, or profits; or business interruption) however
// caused and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of the
// use of this software, even if advised of the possibility of such damage.

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
