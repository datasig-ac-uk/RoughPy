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
// Created by user on 19/10/23.
//

#ifndef ROUGHPY_DEVICE_SRC_CPU_KERNELS_MASKED_UNARY_H_
#define ROUGHPY_DEVICE_SRC_CPU_KERNELS_MASKED_UNARY_H_

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include "devices/types.h"
#include "operators.h"


namespace rpy {
namespace devices {
namespace kernels {

template <typename T, typename Op>
void masked_unary_into_buffer(
        T* RPY_RESTRICT dst,
        const T* RPY_RESTRICT src,
        dimn_t count,
        const bitmask_t* mask,
        Op&& func
) noexcept
{}


extern template void masked_unary_into_buffer<float, identity>(
        float* RPY_RESTRICT,
        const float* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        identity&&);
extern template void masked_unary_into_buffer<float, uminus>(
        float* RPY_RESTRICT,
        const float* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        uminus&&);


extern template void masked_unary_into_buffer<double, identity>(
        double* RPY_RESTRICT,
        const double* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        identity&&);
extern template void masked_unary_into_buffer<double, uminus>(
        double* RPY_RESTRICT,
        const double* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        uminus&&);


extern template void masked_unary_into_buffer<half, identity>(
        half* RPY_RESTRICT,
        const half* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        identity&&);
extern template void masked_unary_into_buffer<half, uminus>(
        half* RPY_RESTRICT,
        const half* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        uminus&&);



extern template void masked_unary_into_buffer<bfloat16, identity>(
        bfloat16* RPY_RESTRICT,
        const bfloat16* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        identity&&);
extern template void masked_unary_into_buffer<bfloat16, uminus>(
        bfloat16* RPY_RESTRICT,
        const bfloat16* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        uminus&&);




extern template void masked_unary_into_buffer<rational_scalar_type, identity>(
        rational_scalar_type* RPY_RESTRICT,
        const rational_scalar_type* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        identity&&);
extern template void masked_unary_into_buffer<rational_scalar_type, uminus>(
        rational_scalar_type* RPY_RESTRICT,
        const rational_scalar_type* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        uminus&&);


extern template void masked_unary_into_buffer<rational_poly_scalar, identity>(
        rational_poly_scalar* RPY_RESTRICT,
        const rational_poly_scalar* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        identity&&);
extern template void masked_unary_into_buffer<rational_poly_scalar, uminus>(
        rational_poly_scalar* RPY_RESTRICT,
        const rational_poly_scalar* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        uminus&&);




}// namespace kernels
}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_SRC_CPU_KERNELS_MASKED_UNARY_H_
