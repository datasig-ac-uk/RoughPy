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


#include "masked_binary.h"

namespace rpy { namespace devices { namespace kernels {

template void masked_binary_into_buffer<rational_poly_scalar, std::plus<rational_poly_scalar>>(
        rational_poly_scalar* RPY_RESTRICT,
        const rational_poly_scalar* RPY_RESTRICT,
        const rational_poly_scalar* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        std::plus<rational_poly_scalar>&&
);
template void masked_binary_into_buffer<rational_poly_scalar, std::minus<rational_poly_scalar>>(
        rational_poly_scalar* RPY_RESTRICT,
        const rational_poly_scalar* RPY_RESTRICT,
        const rational_poly_scalar* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        std::minus<rational_poly_scalar>&&
);
template void masked_binary_into_buffer<rational_poly_scalar, std::multiplies<rational_poly_scalar>>(
        rational_poly_scalar* RPY_RESTRICT,
        const rational_poly_scalar* RPY_RESTRICT,
        const rational_poly_scalar* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        std::multiplies<rational_poly_scalar>&&
);
template void masked_binary_into_buffer<rational_poly_scalar, std::divides<rational_poly_scalar>>(
        rational_poly_scalar* RPY_RESTRICT,
        const rational_poly_scalar* RPY_RESTRICT,
        const rational_poly_scalar* RPY_RESTRICT,
        dimn_t,
        const bitmask_t*,
        std::divides<rational_poly_scalar>&&
);

}}}
