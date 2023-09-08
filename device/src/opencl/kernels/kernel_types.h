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
// Created by user on 04/09/23.
//

#ifndef ROUGHPY_DEVICE_SRC_OPENCL_KERNELS_KERNEL_TYPES_H_
#define ROUGHPY_DEVICE_SRC_OPENCL_KERNELS_KERNEL_TYPES_H_

#define RPY_JOIN(X, Y) X ## Y


#ifdef __OPENCL_C_VERSION__
#  define RPY_KERNEL __kernel
#  define RPY_ADDR_GLOBL __global
#  define RPY_ADDR_LOCAL __local
#  define RPY_ADDR_CONST __constant
#  define RPY_ADDR_PRIVT __private
#  define RPY_ACC_RO __read_only
#  define RPY_ACC_WO __write_only
#  define RPY_ACC_RW __read_write

inline float float_fma(float x, float y, float z) { return fma(x, y, z); }
inline double double_fma(float x, float y, float z) { return fma(x, y, z); }
//inline half half_fma(half x, half y, half z) { return fma(x, y, z); }

#else

#  include "math.h"
#  include "stddef.h"
#  include "stdint.h"

typedef unsigned short ushort;
typedef unsigned uint;
typedef unsigned long ulong;
typedef uint cl_mem_fence_flags;

typedef uint16_t half;

#  define RPY_KERNEL
#  define RPY_ADDR_GLOBL
#  define RPY_ADDR_LOCAL
#  define RPY_ADDR_CONST
#  define RPY_ADDR_PRIVT
#  define RPY_ACC_RO
#  define RPY_ACC_WO
#  define RPY_ACC_RW
#  define CLK_GLOBAL_MEM_FENCE 0
#  define CLK_LOCAL_MEM_FENCE 0
#  define CLK_IMAGE_MEM_FENCE 0

extern uint get_work_dim();
extern size_t get_global_size(uint);
extern size_t get_global_id(uint);
extern size_t get_local_size(uint);
extern size_t get_enqueued_local_size(uint);
extern size_t get_local_id(uint);
extern size_t get_num_groups(uint);
extern size_t get_group_id(uint);
extern size_t get_global_offset(uint);
extern size_t get_global_linear_id();
extern size_t get_local_linear_id();

inline float float_fma(float x, float y, float z) { return fmaf(x, y, z); }
inline double double_fma(float x, float y, float z) { return fma(x, y, z); }
//extern half half_fma(half, half, half);



extern int max(int, int);
extern int min(int, int);


extern void barrier(cl_mem_fence_flags);
extern void work_group_barrier(cl_mem_fence_flags);




#endif

typedef unsigned mask_kt;
typedef int deg_kt;
typedef long dimn_kt;

#define RPY_MASK_ELT_SIZE (sizeof(mask_kt) * 8)

#endif// ROUGHPY_DEVICE_SRC_OPENCL_KERNELS_KERNEL_TYPES_H_
