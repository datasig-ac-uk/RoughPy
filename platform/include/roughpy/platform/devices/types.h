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

#ifndef ROUGHPY_DEVICE_TYPES_H_
#define ROUGHPY_DEVICE_TYPES_H_

#include <complex>

// Maybe replace this, Eigen/Core brings in so much stuff.
#include <Eigen/Core>
#include <libalgebra_lite/coefficients.h>
#include <libalgebra_lite/packed_integer.h>
#include <libalgebra_lite/polynomial.h>
#include <libalgebra_lite/polynomial_basis.h>

#include "core.h"

#ifdef LAL_NO_USE_GMP
#  define RPY_USING_GMP 0
#else
#  define RPY_USING_GMP 1
#endif

namespace rpy {
namespace devices {


/// half-precision complex float
using half_complex = std::complex<half>;

/// BFloat16 complex float - probably not supported anywhere
using bf16_complex = std::complex<bfloat16>;

/// Single precision complex float
using float_complex = std::complex<float>;

/// double precision complex float
using double_complex = std::complex<double>;

/// Monomial key-type of polynomials
using monomial = lal::monomial;

/// Indeterminate type for monomials
using indeterminate_type = typename monomial::letter_type;




}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_TYPES_H_
