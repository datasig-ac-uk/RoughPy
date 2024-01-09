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

#ifndef ROUGHPY_DEVICE_SRC_CPU_KERNELS_MASKED_BINARY_H_
#define ROUGHPY_DEVICE_SRC_CPU_KERNELS_MASKED_BINARY_H_

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include "devices/types.h"
#include <functional>

namespace rpy {
namespace devices {
namespace kernels {

template <typename T, typename Op>
void masked_binary_into_buffer(
        T* RPY_RESTRICT dst,
        const T* RPY_RESTRICT lhs,
        const T* RPY_RESTRICT rhs,
        dimn_t count,
        const bitmask_t* mask,
        Op&& func
) noexcept
{}

extern template void masked_binary_into_buffer<float, std::plus<float>>(
        float* RPY_RESTRICT,
        const float* RPY_RESTRICT,
        const float* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        std::plus<float>&&
);
extern template void masked_binary_into_buffer<float, std::minus<float>>(
        float* RPY_RESTRICT,
        const float* RPY_RESTRICT,
        const float* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        std::minus<float>&&
);
extern template void masked_binary_into_buffer<float, std::multiplies<float>>(
        float* RPY_RESTRICT,
        const float* RPY_RESTRICT,
        const float* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        std::multiplies<float>&&
);
extern template void masked_binary_into_buffer<float, std::divides<float>>(
        float* RPY_RESTRICT,
        const float* RPY_RESTRICT,
        const float* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        std::divides<float>&&
);

extern template void masked_binary_into_buffer<double, std::plus<double>>(
        double* RPY_RESTRICT,
        const double* RPY_RESTRICT,
        const double* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        std::plus<double>&&
);
extern template void masked_binary_into_buffer<double, std::minus<double>>(
        double* RPY_RESTRICT,
        const double* RPY_RESTRICT,
        const double* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        std::minus<double>&&
);
extern template void masked_binary_into_buffer<double, std::multiplies<double>>(
        double* RPY_RESTRICT,
        const double* RPY_RESTRICT,
        const double* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        std::multiplies<double>&&
);
extern template void masked_binary_into_buffer<double, std::divides<double>>(
        double* RPY_RESTRICT,
        const double* RPY_RESTRICT,
        const double* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        std::divides<double>&&
);

extern template void masked_binary_into_buffer<half, std::plus<half>>(
        half* RPY_RESTRICT,
        const half* RPY_RESTRICT,
        const half* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        std::plus<half>&&
);
extern template void masked_binary_into_buffer<half, std::minus<half>>(
        half* RPY_RESTRICT,
        const half* RPY_RESTRICT,
        const half* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        std::minus<half>&&
);
extern template void masked_binary_into_buffer<half, std::multiplies<half>>(
        half* RPY_RESTRICT,
        const half* RPY_RESTRICT,
        const half* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        std::multiplies<half>&&
);
extern template void masked_binary_into_buffer<half, std::divides<half>>(
        half* RPY_RESTRICT,
        const half* RPY_RESTRICT,
        const half* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        std::divides<half>&&
);

extern template void masked_binary_into_buffer<bfloat16, std::plus<bfloat16>>(
        bfloat16* RPY_RESTRICT,
        const bfloat16* RPY_RESTRICT,
        const bfloat16* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        std::plus<bfloat16>&&
);
extern template void masked_binary_into_buffer<bfloat16, std::minus<bfloat16>>(
        bfloat16* RPY_RESTRICT,
        const bfloat16* RPY_RESTRICT,
        const bfloat16* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        std::minus<bfloat16>&&
);
extern template void masked_binary_into_buffer<bfloat16, std::multiplies<bfloat16>>(
        bfloat16* RPY_RESTRICT,
        const bfloat16* RPY_RESTRICT,
        const bfloat16* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        std::multiplies<bfloat16>&&
);
extern template void masked_binary_into_buffer<bfloat16, std::divides<bfloat16>>(
        bfloat16* RPY_RESTRICT,
        const bfloat16* RPY_RESTRICT,
        const bfloat16* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        std::divides<bfloat16>&&
);

extern template void masked_binary_into_buffer<rational_scalar_type, std::plus<rational_scalar_type>>(
        rational_scalar_type* RPY_RESTRICT,
        const rational_scalar_type* RPY_RESTRICT,
        const rational_scalar_type* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        std::plus<rational_scalar_type>&&
);
extern template void masked_binary_into_buffer<rational_scalar_type, std::minus<rational_scalar_type>>(
        rational_scalar_type* RPY_RESTRICT,
        const rational_scalar_type* RPY_RESTRICT,
        const rational_scalar_type* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        std::minus<rational_scalar_type>&&
);
extern template void masked_binary_into_buffer<rational_scalar_type, std::multiplies<rational_scalar_type>>(
        rational_scalar_type* RPY_RESTRICT,
        const rational_scalar_type* RPY_RESTRICT,
        const rational_scalar_type* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        std::multiplies<rational_scalar_type>&&
);
extern template void masked_binary_into_buffer<rational_scalar_type, std::divides<rational_scalar_type>>(
        rational_scalar_type* RPY_RESTRICT,
        const rational_scalar_type* RPY_RESTRICT,
        const rational_scalar_type* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        std::divides<rational_scalar_type>&&
);


extern template void masked_binary_into_buffer<rational_poly_scalar, std::plus<rational_poly_scalar>>(
        rational_poly_scalar* RPY_RESTRICT,
        const rational_poly_scalar* RPY_RESTRICT,
        const rational_poly_scalar* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        std::plus<rational_poly_scalar>&&
);
extern template void masked_binary_into_buffer<rational_poly_scalar, std::minus<rational_poly_scalar>>(
        rational_poly_scalar* RPY_RESTRICT,
        const rational_poly_scalar* RPY_RESTRICT,
        const rational_poly_scalar* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        std::minus<rational_poly_scalar>&&
);
extern template void masked_binary_into_buffer<rational_poly_scalar, std::multiplies<rational_poly_scalar>>(
        rational_poly_scalar* RPY_RESTRICT,
        const rational_poly_scalar* RPY_RESTRICT,
        const rational_poly_scalar* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        std::multiplies<rational_poly_scalar>&&
);
extern template void masked_binary_into_buffer<rational_poly_scalar, std::divides<rational_poly_scalar>>(
        rational_poly_scalar* RPY_RESTRICT,
        const rational_poly_scalar* RPY_RESTRICT,
        const rational_poly_scalar* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        std::divides<rational_poly_scalar>&&
);



}// namespace kernels
}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_SRC_CPU_KERNELS_MASKED_BINARY_H_
