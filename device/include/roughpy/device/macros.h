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
// Created by user on 24/08/23.
//

#ifndef ROUGHPY_DEVICE_MACROS_H_
#define ROUGHPY_DEVICE_MACROS_H_

#if defined(__NVCC__)
#  include <cuda.h>

#  define RPY_DEVICE __device__
#  define RPY_HOST __host__
#  define RPY_DEVICE_HOST __device__ __host__
#  define RPY_KERNEL __global__
#  define RPY_DEVICE_SHARED __shared__
#  define RPY_STRONG_INLINE __inline__

#elif defined(__HIPCC__)

#  define RPY_DEVICE __device__
#  define RPY_HOST __host__
#  define RPY_DEVICE_HOST __device__ __host__
#  define RPY_KERNEL __global__
#  define RPY_DEVICE_SHARED __shared__
#  define RPY_STRONG_INLINE

#else
#  define RPY_DEVICE
#  define RPY_HOST
#  define RPY_DEVICE_HOST
#  define RPY_KERNEL
#  define RPY_DEVICE_SHARED
#  define RPY_STRONG_INLINE

#endif

#endif// ROUGHPY_DEVICE_MACROS_H_
