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
// Created by user on 02/08/23.
//

#ifndef ROUGHPY_SCALARS_SRC_LINEAR_ALGEBRA_BLAS_H_
#define ROUGHPY_SCALARS_SRC_LINEAR_ALGEBRA_BLAS_H_

#include "scalar_blas_defs.h"

#include <roughpy/scalars/types.h>

#ifndef ROUGHPY_DISABLE_BLAS
#define RPY_BLAS_FUNC_(NAME) RPY_JOIN(cblas_, NAME)
#define RPY_BLAS_FUNC(NAME) RPY_BLAS_FUNC_(RPY_JOIN(RPY_BLA_SPRX, NAME))

#define RPY_BLA_LYO_ARG(ARG) ARG
#define RPY_BLA_TPS_ARG(ARG) ARG
#define RPY_BLA_INT_ARG(ARG) ARG
#define RPY_BLA_SCA_ARG(ARG) convert_scalar(ARG)
#define RPY_BLA_PTR_ARG(ARG) \
    reinterpret_cast<bla_scalar*>(ARG)
#define RPY_BLA_CPT_ARG(ARG) \
    reinterpret_cast<const bla_scalar*>(ARG)

#define RPY_BLA_CALL_(ROUTINE, ...) ROUTINE(__VA_ARGS__)
#define RPY_BLA_CALL(ROUTINE, ...) RPY_BLAS_FUNC(ROUTINE)(__VA_ARGS__)

namespace rpy {
namespace scalars {
namespace blas {

using ::rpy::blas::BlasLayout;
using ::rpy::blas::BlasTranspose;
using ::rpy::blas::BlasUpLo;

namespace dtl {
template <typename S>
struct BlasScaConversion {
    using type = S;
};
template <>
struct BlasScaConversion<float_complex> {
    using type = rpy::blas::complex32;
};
template <>
struct BlasScaConversion<double_complex> {
    using type = rpy::blas::complex64;
};
}

template <typename S>
using BlasScalar = typename dtl::BlasScaConversion<S>::type;


template <typename S, typename R>
struct blas_funcs {
    using integer = rpy::blas::integer;
    using logical = rpy::blas::logical;

    using scalar = S;
    using abs_scalar = R;
    using bla_scalar = BlasScalar<S>;

    constexpr BlasLayout to_blas_layout(MatrixLayout layout) {
        return layout == rpy::scalars::MatrixLayout::RowMajor
                ? BlasLayout::CblasRowMajor
                : BlasLayout::CblasColMajor;
    }

    // Level 1 functions
    static void
    axpy(const integer n, const scalar& alpha, const scalar* RPY_RESTRICT x,
         const integer incx, scalar* RPY_RESTRICT y,
         const integer incy) noexcept;

    static scalar
    dot(const integer n, const scalar* RPY_RESTRICT x, const integer incx,
        const scalar* RPY_RESTRICT y, const integer incy) noexcept;

    static abs_scalar
    asum(const integer n, const scalar* RPY_RESTRICT x,
         const integer incx) noexcept;

    static abs_scalar
    nrm2(const integer n, const scalar* RPY_RESTRICT x,
         const integer incx) noexcept;

    static integer
    iamax(const integer n, const scalar* RPY_RESTRICT x,
          const integer incx) noexcept;

    // Level 2
    static void
    gemv(BlasLayout layout, BlasTranspose trans, const integer m,
         const integer n, const scalar& alpha, const scalar* RPY_RESTRICT A,
         const integer lda, const scalar* RPY_RESTRICT x, const integer incx,
         const scalar& beta, scalar* RPY_RESTRICT y,
         const integer incy) noexcept;

    // Level 3
    static void
    gemm(BlasLayout layout, BlasTranspose transa, BlasTranspose transb,
         const integer m, const integer n, const integer k, const scalar& alpha,
         const scalar* RPY_RESTRICT A, const integer lda,
         const scalar* RPY_RESTRICT B, const integer ldb, const scalar& beta,
         scalar* RPY_RESTRICT C, const integer ldc) noexcept;
};

}// namespace blas
}// namespace scalars
}// namespace rpy
#endif
#endif// ROUGHPY_SCALARS_SRC_LINEAR_ALGEBRA_BLAS_H_
