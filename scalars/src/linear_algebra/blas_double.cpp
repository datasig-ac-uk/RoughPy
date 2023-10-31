// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
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
// Created by user on 02/08/23.
//

#ifndef ROUGHPY_SCALARS_SRC_LINEAR_ALGEBRA_BLAS_DOUBLE_CPP_
#define ROUGHPY_SCALARS_SRC_LINEAR_ALGEBRA_BLAS_DOUBLE_CPP_

#include "blas.h"

#ifndef ROUGHPY_DISABLE_BLAS
#  define RPY_BLA_SPRX d

static constexpr double convert_scalar(const double& arg) noexcept
{
    return arg;
}

namespace rpy {
namespace scalars {
namespace blas {

template <>
void blas_funcs<double, double>::axpy(
        const integer n, const scalar& alpha, const scalar* x,
        const integer incx, scalar* y, const integer incy
) noexcept
{
    RPY_BLA_CALL(
            axpy, RPY_BLA_INT_ARG(n), RPY_BLA_SCA_ARG(alpha),
            RPY_BLA_CPT_ARG(x), RPY_BLA_INT_ARG(incx), RPY_BLA_PTR_ARG(y),
            RPY_BLA_INT_ARG(incy)
    );
}
template <>
typename blas_funcs<double, double>::scalar blas_funcs<double, double>::dot(
        const integer n, const scalar* x, const integer incx, const scalar* y,
        const integer incy
) noexcept
{
    return RPY_BLA_CALL(
            dot, RPY_BLA_INT_ARG(n), RPY_BLA_CPT_ARG(x), RPY_BLA_INT_ARG(incx),
            RPY_BLA_CPT_ARG(y), RPY_BLA_INT_ARG(incy)
    );
}
template <>
typename blas_funcs<double, double>::abs_scalar
blas_funcs<double, double>::asum(
        const integer n, const scalar* x, const integer incx
) noexcept
{
    return RPY_BLA_CALL(
            asum, RPY_BLA_INT_ARG(n), RPY_BLA_CPT_ARG(x), RPY_BLA_INT_ARG(incx)
    );
}
template <>
typename blas_funcs<double, double>::abs_scalar
blas_funcs<double, double>::nrm2(
        const integer n, const scalar* x, const integer incx
) noexcept
{
    return RPY_BLA_CALL(
            nrm2, RPY_BLA_INT_ARG(n), RPY_BLA_CPT_ARG(x), RPY_BLA_INT_ARG(incx)
    );
}
template <>
typename blas_funcs<double, double>::integer blas_funcs<double, double>::iamax(
        const integer n, const scalar* x, const integer incx
) noexcept
{
    return RPY_BLA_CALL_(
            RPY_BLAS_FUNC_(RPY_JOIN(RPY_JOIN(i, RPY_BLA_SPRX), amax)),
            RPY_BLA_INT_ARG(n), RPY_BLA_CPT_ARG(x), RPY_BLA_INT_ARG(incx)
    );
}
template <>
void blas_funcs<double, double>::gemv(
        BlasLayout layout, BlasTranspose trans, const integer m,
        const integer n, const scalar& alpha, const scalar* A,
        const integer lda, const scalar* x, const integer incx,
        const scalar& beta, scalar* y, const integer incy
) noexcept
{
    RPY_BLA_CALL(
            gemv, RPY_BLA_LYO_ARG(layout), RPY_BLA_TPS_ARG(trans),
            RPY_BLA_INT_ARG(m), RPY_BLA_INT_ARG(n), RPY_BLA_SCA_ARG(alpha),
            RPY_BLA_CPT_ARG(A), RPY_BLA_INT_ARG(lda), RPY_BLA_CPT_ARG(x),
            RPY_BLA_INT_ARG(incx), RPY_BLA_SCA_ARG(beta), RPY_BLA_PTR_ARG(y),
            RPY_BLA_INT_ARG(incy)
    );
}
template <>
void blas_funcs<double, double>::gemm(
        BlasLayout layout, BlasTranspose transa, BlasTranspose transb,
        const integer m, const integer n, const integer k, const scalar& alpha,
        const scalar* A, const integer lda, const scalar* B, const integer ldb,
        const scalar& beta, scalar* C, const integer ldc
) noexcept
{
    RPY_BLA_CALL(
            gemm, RPY_BLA_LYO_ARG(layout), RPY_BLA_TPS_ARG(transa),
            RPY_BLA_TPS_ARG(transb), RPY_BLA_INT_ARG(m), RPY_BLA_INT_ARG(n),
            RPY_BLA_INT_ARG(k), RPY_BLA_SCA_ARG(alpha), RPY_BLA_CPT_ARG(A),
            RPY_BLA_INT_ARG(lda), RPY_BLA_CPT_ARG(B), RPY_BLA_INT_ARG(ldb),
            RPY_BLA_SCA_ARG(beta), RPY_BLA_PTR_ARG(C), RPY_BLA_INT_ARG(ldc)
    );
}

}// namespace blas
}// namespace scalars
}// namespace rpy

#endif
#endif// ROUGHPY_SCALARS_SRC_LINEAR_ALGEBRA_BLAS_DOUBLE_CPP_
