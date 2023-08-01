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

#include "scalar_blas_defs.h"
#include <roughpy/scalars/scalars_fwd.h>

#include <roughpy/scalars/scalar_blas.h>
#include <roughpy/scalars/scalar_array.h>
#include <roughpy/scalars/owned_scalar_array.h>
#include <roughpy/scalars/scalar.h>

#define RPY_BLAS_FUNC(NAME) NAME
#define RPY_LAPACK_FUNC(NAME) NAME

namespace rpy {
namespace blas {

template <typename S, typename R>
struct blas_funcs {
    using scalar = S;
    using abs_scalar = R;

    // Level 1 functions
    inline static void
    axpy(const integer n, const scalar& alpha, const scalar* RPY_RESTRICT x,
         const integer incx, scalar* RPY_RESTRICT y,
         const integer incy) noexcept;

    inline static scalar
    dot(const integer n, const scalar* RPY_RESTRICT x, const integer incx,
        const scalar* RPY_RESTRICT y, const integer incy) noexcept;

    inline static abs_scalar
    asum(const integer n, const scalar* RPY_RESTRICT x,
         const integer incx) noexcept;

    inline static abs_scalar
    nrm2(const integer n, const scalar* RPY_RESTRICT x,
         const integer incx) noexcept;

    inline static integer
    iamax(const integer n, const scalar* RPY_RESTRICT x,
          const integer incx) noexcept;

    // Level 2

    inline static void
    gemv(BlasTranspose trans, const integer m, const integer n,
         const scalar& alpha, const scalar* RPY_RESTRICT A, const integer lda,
         const scalar* RPY_RESTRICT x, const integer incx, const scalar& beta,
         scalar* RPY_RESTRICT y, const integer incy) noexcept;

    // Level 3

    inline static void
    gemm(BlasTranspose transa, BlasTranspose transb, const integer m,
         const integer n, const integer k, const scalar& alpha,
         const scalar* RPY_RESTRICT A, const integer lda,
         const scalar* RPY_RESTRICT B, const integer ldb, const scalar& beta,
         scalar* RPY_RESTRICT C, const integer ldc) noexcept;
};

template <>
struct blas_funcs<scalars::float_complex, double> {
    using scalar = scalars::float_complex;
    using abs_scalar = float;

    // Level 1 functions
    inline static void
    axpy(const integer n, const scalar& alpha, const scalar* RPY_RESTRICT x,
         const integer incx, scalar* RPY_RESTRICT y,
         const integer incy) noexcept
    {}

    inline static scalar
    dot(const integer n, const scalar* RPY_RESTRICT x, const integer incx,
        const scalar* RPY_RESTRICT y, const integer incy) noexcept
    {}

    inline static abs_scalar
    asum(const integer n, const scalar* RPY_RESTRICT x,
         const integer incx) noexcept
    {}

    inline static abs_scalar
    nrm2(const integer n, const scalar* RPY_RESTRICT x,
         const integer incx) noexcept
    {}

    inline static integer
    iamax(const integer n, const scalar* RPY_RESTRICT x,
          const integer incx) noexcept
    {}

    // Level 2

    inline static void
    gemv(BlasTranspose trans, const integer m, const integer n,
         const scalar& alpha, const scalar* RPY_RESTRICT A, const integer lda,
         const scalar* RPY_RESTRICT x, const integer incx, const scalar& beta,
         scalar* RPY_RESTRICT y, const integer incy) noexcept
    {}

    // Level 3

    inline static void
    gemm(BlasTranspose transa, BlasTranspose transb, const integer m,
         const integer n, const integer k, const scalar& alpha,
         const scalar* RPY_RESTRICT A, const integer lda,
         const scalar* RPY_RESTRICT B, const integer ldb, const scalar& beta,
         scalar* RPY_RESTRICT C, const integer ldc)
    {}
};

}// namespace blas

namespace lapack {
using blas::complex32;
using blas::complex64;
using blas::integer;
using blas::logical;

template <typename S, typename R>
struct lapack_func_workspace;

template <typename S, typename R>
struct lapack_funcs : lapack_func_workspace<S, R> {
    using scalar = S;
    using real_scalar = R;

    static inline void
    gesv(const integer n, const integer nrhs, scalar* RPY_RESTRICT A,
         const integer lda, integer* RPY_RESTRICT ipiv, scalar* RPY_RESTRICT B,
         const integer ldb, integer& info) noexcept;

    static inline void
    syev(const char* jobz, blas::BlasUpLo uplo, const integer n,
         scalar* RPY_RESTRICT A, const integer lda, real_scalar* RPY_RESTRICT w,
         scalar* RPY_RESTRICT work, const integer* RPY_RESTRICT lwork,
         real_scalar* RPY_RESTRICT rwork, integer& info) noexcept;

    static inline void
    geev(const char* joblv, const char* jobvr, const integer n,
         scalar* RPY_RESTRICT A, const integer lda, scalar* RPY_RESTRICT wr,
         scalar* RPY_RESTRICT RPY_UNUSED_VAR wi, scalar* RPY_RESTRICT vl,
         const integer ldvl, scalar* RPY_RESTRICT vr, const integer ldvr,
         scalar* RPY_RESTRICT work, const integer* RPY_RESTRICT lwork,
         real_scalar* RPY_RESTRICT rwork, integer& info) noexcept;

    static inline void
    gesvd(const char* jobu, const char* jobvt, const integer m, const integer n,
          scalar* RPY_RESTRICT A, const integer lda,
          real_scalar* RPY_RESTRICT s, scalar* RPY_RESTRICT u,
          const integer ldu, scalar* RPY_RESTRICT vt, const integer ldvt,
          scalar* RPY_RESTRICT work, const integer* RPY_RESTRICT lwork,
          real_scalar* RPY_UNUSED_VAR rwork, integer& info) noexcept;

    static inline void
    gesdd(const char* jobz, const integer m, const integer n,
          scalar* RPY_RESTRICT A, const integer lda,
          real_scalar* RPY_RESTRICT s, scalar* RPY_RESTRICT u,
          const integer ldu, scalar* RPY_RESTRICT vt, const integer ldvt,
          scalar* RPY_RESTRICT work, const integer* RPY_RESTRICT lwork,
          integer* RPY_RESTRICT iwork, real_scalar* RPY_RESTRICT rwork,
          integer& info) noexcept;

    static inline void
    gels(blas::BlasTranspose trans, const integer m, const integer n,
         const integer nrhs, scalar* RPY_RESTRICT A, const integer lda,
         scalar* RPY_RESTRICT B, const integer ldb, scalar* RPY_RESTRICT work,
         const integer* RPY_RESTRICT lwork, integer& info) noexcept;

    static inline void
    gelsy(const integer m, const integer n, const integer nrhs,
          scalar* RPY_RESTRICT A, const integer lda, scalar* RPY_RESTRICT B,
          const integer ldb, integer* RPY_RESTRICT jpvt,
          const real_scalar& rcond, integer& rank, scalar* RPY_RESTRICT work,
          const integer* RPY_RESTRICT lwork, real_scalar* RPY_RESTRICT rwork,
          integer& info) noexcept;

    static inline void
    gelss(const integer m, const integer n, const integer nrhs,
          scalar* RPY_RESTRICT A, const integer lda, scalar* RPY_RESTRICT B,
          const integer ldb, real_scalar* RPY_RESTRICT s,
          const real_scalar& rcond, integer& rank, scalar* RPY_RESTRICT work,
          const integer* RPY_RESTRICT lwork, real_scalar* RPY_RESTRICT rwork,
          integer& info) noexcept;

    static inline void
    gelsd(const integer m, const integer n, const integer nrhs,
          scalar* RPY_RESTRICT A, const integer lda, scalar* RPY_RESTRICT B,
          const integer ldb, real_scalar* RPY_RESTRICT s,
          const real_scalar& rcond, integer& rank, scalar* RPY_RESTRICT work,
          const integer* RPY_RESTRICT lwork, real_scalar* RPY_RESTRICT rwork,
          integer* RPY_RESTRICT iwork, integer& info) noexcept;
};

}// namespace lapack

namespace scalars {

template <typename S, typename R>
class StandardLinearAlgebra : public BlasInterface,
                              blas::blas_funcs<S, R>,
                              lapack::lapack_funcs<S, R>
{

    using blas = ::rpy::blas::blas_funcs<S, R>;
    using lapack  = ::rpy::lapack::lapack_funcs<S, R>;

    using integer = ::rpy::blas::integer;
    using logical = ::rpy::blas::logical;

    using typename blas::scalar;
    using typename blas::abs_scalar;
    using typename lapack::real_scalar;

    using BlasLayout = ::rpy::blas::BlasLayout;
    using BlasTranspose = ::rpy::blas::BlasTranspose;
    using BlasUpLo = ::rpy::blas::BlasUpLo;
    using BlasDiag = ::rpy::blas::BlasDiag;
    using BlasSide = ::rpy::blas::BlasSide;

    static constexpr auto Blas_ColMajor = BlasLayout::CblasColMajor;
    static constexpr auto Blas_RowMajor = BlasLayout::CblasRowMajor;

    static constexpr auto Blas_NoTrans = BlasTranspose::CblasNoTrans;
    static constexpr auto Blas_Trans = BlasTranspose::CblasTrans;

    static constexpr auto Blas_Up = BlasUpLo::CblasUpper;
    static constexpr auto Blas_Lo = BlasUpLo::CblasLower;

    static constexpr auto Blas_DUnit = BlasDiag::CblasUnit;
    static constexpr auto Blas_DNoUnit = BlasDiag::CblasNonUnit;

    static constexpr auto Blas_Left = BlasSide::CblasLeft;
    static constexpr auto Blas_Right = BlasSide::CblasRight;

public:
    void transpose(ScalarMatrix& matrix) const override
    {
        BlasInterface::transpose(matrix);
    }
    OwnedScalarArray vector_axpy(
            const ScalarArray& x, const Scalar& a, const ScalarArray& y
    ) override
    {
        auto guard = lock();
        const auto* type = BlasInterface::type();
        RPY_CHECK(x.type() == type && y.type() == type);
        OwnedScalarArray result(type, y.size());
        type->convert_copy(result, y, y.size());

        auto N = static_cast<integer>(y.size());
        const auto alpha = scalar_cast<float>(a);

        blas::axpy(
                N, alpha, x.raw_cast<const float*>(), 1,
                result.raw_cast<float*>(), 1
        );
        return result;
    }
    Scalar dot_product(const ScalarArray& lhs, const ScalarArray& rhs) override
    {
        auto guard = lock();
        const auto* type = BlasInterface::type();

        RPY_CHECK(lhs.type() == type && rhs.type() == type);

        auto N = static_cast<integer>(lhs.size());
        auto result = blas::dot(
                N, lhs.raw_cast<const float*>(), 1, rhs.raw_cast<const float*>(), 1
        );
        return {type, result};
    }
    Scalar L1Norm(const ScalarArray& vector) override
    {
        auto guard = lock();
        auto N = static_cast<integer>(vector.size());
        auto result = blas::asum(N, vector.raw_cast<const float*>(), 1);
        return {type(), result};
    }
    Scalar L2Norm(const ScalarArray& vector) override
    {
        RPY_CHECK(vector.type() == type());
        float result = 0.0;
        auto N = static_cast<integer>(vector.size());
        result = blas::nrm2(N, vector.raw_cast<const float*>(), 1);
        return {type(), result};
    }
    Scalar LInfNorm(const ScalarArray& vector) override
    {
        RPY_CHECK(vector.type() == type());
        auto N = static_cast<integer>(vector.size());
        const auto* ptr = vector.raw_cast<const float*>();
        auto idx = blas::iamax(N, ptr, 1);
        auto result = ptr[idx];
        return {type(), result};
    }
    void
    gemv(ScalarMatrix& y, const ScalarMatrix& A, const ScalarMatrix& x,
         const Scalar& alpha, const Scalar& beta) override
    {

        auto guard = lock();
        type_check(A);
        type_check(x);
        const float alp = scalar_cast<float>(alpha);
        const float bet = scalar_cast<float>(beta);

        integer m = A.nrows();
        integer n = A.ncols();
        integer lda = A.leading_dimension();

        integer incx;
        integer n_eqns;
        if (x.layout() == MatrixLayout::RowMajor) {
            incx = x.ncols();
            n_eqns = x.nrows();
        } else {
            incx = x.nrows();
            n_eqns = x.ncols();
        }

        /*
     * We're assuming that the vectors are stored contiguously so that incx
     * is simply the number of columns/rows depending on whether it is row
     * major or column major.
         */
        matrix_product_check(n, incx);

        /*
     * For now, we shall assume that we don't want to transpose the matrix A.
     * In the future we might want to consider transposing based on whether
     * the matrix is in row major or column major format.
         */
        BlasTranspose transa = blas::Blas_NoTrans;

        BlasLayout layout = A.layout() == MatrixLayout::RowMajor
                                  ? blas::Blas_RowMajor
                                  : blas::Blas_ColMajor;

        integer incy;
        if (y.type() == nullptr || y.is_null()) {
            /*
         * Type is null, so it hasn't been
         * initialized yet. Allocate a new empty matrix of the correct size.
             */
            y = ScalarMatrix(type(), m, n_eqns, MatrixLayout::ColumnMajor);
            incy = m;
        } else {
            type_check(y);
            integer n_eqs_check;
            if (y.layout() == MatrixLayout::RowMajor) {
                incy = y.ncols();
                n_eqs_check = y.nrows();
            } else {
                incy = y.nrows();
                n_eqs_check = y.ncols();
            }

            RPY_CHECK(incy == incx && n_eqs_check == n_eqns);
        }

        cblas_sgemv(
                layout, transa, m, n, alp, A.raw_cast<const float*>(), lda,
                x.raw_cast<const float*>(), incx, bet, y.raw_cast<float*>(), incy
        );
    }
    void
    gemm(ScalarMatrix& C, const ScalarMatrix& A, const ScalarMatrix& B,
         const Scalar& alpha, const Scalar& beta) override
    {
    }
    void gesv(ScalarMatrix& A, ScalarMatrix& B) override
    {
    }
    EigenDecomposition syev(ScalarMatrix& A, bool eigenvectors) override
    {
    }
    EigenDecomposition gees(ScalarMatrix& A, bool eigenvectors) override
    {
    }
    SingularValueDecomposition
    gesvd(ScalarMatrix& A, bool return_U, bool return_VT) override
    {
    }
    SingularValueDecomposition
    gesdd(ScalarMatrix& A, bool return_U, bool return_VT) override
    {
    }
    void gels(ScalarMatrix& A, ScalarMatrix& b) override
    {
    }
    ScalarMatrix gelsy(ScalarMatrix& A, ScalarMatrix& b) override
    {
    }
    ScalarMatrix gelss(ScalarMatrix& A, ScalarMatrix& b) override
    {
    }
    ScalarMatrix gelsd(ScalarMatrix& A, ScalarMatrix& b) override
    {
    }
};

}// namespace scalars

namespace blas {

template <>
void blas_funcs<float, float>::axpy(
        const integer n, const scalar& alpha, const scalar* x,
        const integer incx, scalar* y, const integer incy
) noexcept
{
    RPY_BLAS_FUNC(saxpy)(&n, &alpha, x, &incx, y, &incy);
}
template <>
typename blas_funcs<float, float>::scalar blas_funcs<float, float>::dot(
        const integer n, const scalar* x, const integer incx, const scalar* y,
        const integer incy
) noexcept
{
    return RPY_BLAS_FUNC(sdot)(&n, x, &incx, y, &incy);
}
template <>
typename blas_funcs<float, float>::abs_scalar blas_funcs<float, float>::asum(
        const integer n, const scalar* x, const integer incx
) noexcept
{
    return RPY_BLAS_FUNC(sasum)(&n, x, &incx);
}
template <>
typename blas_funcs<float, float>::abs_scalar blas_funcs<float, float>::nrm2(
        const integer n, const scalar* x, const integer incx
) noexcept
{
    return RPY_BLAS_FUNC(snrm2)(&n, x, &incx);
}
template <>
integer blas_funcs<float, float>::iamax(
        const integer n, const scalar* x, const integer incx
) noexcept
{
    return RPY_BLAS_FUNC(isamax)(&n, x, &incx);
}
template <>
void blas_funcs<float, float>::gemv(
        BlasTranspose trans, const integer m, const integer n,
        const scalar& alpha, const scalar* A, const integer lda,
        const scalar* x, const integer incx, const scalar& beta, scalar* y,
        const integer incy
) noexcept
{
    RPY_BLAS_FUNC(sgemv)
    (reinterpret_cast<const char*>(&trans), &m, &n, &alpha, A, &lda, x, &incx,
     &beta, y, &incy);
}
template <>
void blas_funcs<float, float>::gemm(
        BlasTranspose transa, BlasTranspose transb, const integer m,
        const integer n, const integer k, const scalar& alpha, const scalar* A,
        const integer lda, const scalar* B, const integer ldb,
        const scalar& beta, scalar* C, const integer ldc
) noexcept
{
    RPY_BLAS_FUNC(sgemm)
    (reinterpret_cast<const char*>(&transa),
     reinterpret_cast<const char*>(&transb), &m, &n, &k, &alpha, A, &lda, B,
     &ldb, &beta, C, &ldc);
}

template <>
void blas_funcs<double, double>::axpy(
        const integer n, const scalar& alpha, const scalar* x,
        const integer incx, scalar* y, const integer incy
) noexcept
{
    RPY_BLAS_FUNC(daxpy)(&n, &alpha, x, &incx, y, &incy);
}
template <>
typename blas_funcs<double, double>::scalar blas_funcs<double, double>::dot(
        const integer n, const scalar* x, const integer incx, const scalar* y,
        const integer incy
) noexcept
{
    return RPY_BLAS_FUNC(ddot)(&n, x, &incx, y, &incy);
}
template <>
typename blas_funcs<double, double>::abs_scalar
blas_funcs<double, double>::asum(
        const integer n, const scalar* x, const integer incx
) noexcept
{
    return RPY_BLAS_FUNC(dasum)(&n, x, &incx);
}
template <>
typename blas_funcs<double, double>::abs_scalar
blas_funcs<double, double>::nrm2(
        const integer n, const scalar* x, const integer incx
) noexcept
{
    return RPY_BLAS_FUNC(dnrm2)(&n, x, &incx);
}
template <>
integer blas_funcs<double, double>::iamax(
        const integer n, const scalar* x, const integer incx
) noexcept
{
    return RPY_BLAS_FUNC(idamax)(&n, x, &incx);
}
template <>
void blas_funcs<double, double>::gemv(
        BlasTranspose trans, const integer m, const integer n,
        const scalar& alpha, const scalar* A, const integer lda,
        const scalar* x, const integer incx, const scalar& beta, scalar* y,
        const integer incy
) noexcept
{
    RPY_BLAS_FUNC(dgemv)
    (reinterpret_cast<const char*>(&trans), &m, &n, &alpha, A, &lda, x, &incx,
     &beta, y, &incy);
}
template <>
void blas_funcs<double, double>::gemm(
        BlasTranspose transa, BlasTranspose transb, const integer m,
        const integer n, const integer k, const scalar& alpha, const scalar* A,
        const integer lda, const scalar* B, const integer ldb,
        const scalar& beta, scalar* C, const integer ldc
) noexcept
{
    RPY_BLAS_FUNC(dgemm)
    (reinterpret_cast<const char*>(&transa),
     reinterpret_cast<const char*>(&transb), &m, &n, &k, &alpha, A, &lda, B,
     &ldb, &beta, C, &ldc);
}

template <>
void blas_funcs<scalars::float_complex, float>::axpy(
        const integer n, const scalar& alpha, const scalar* x,
        const integer incx, scalar* y, const integer incy
) noexcept
{
    RPY_BLAS_FUNC(caxpy)
    (&n, reinterpret_cast<const complex32*>(&alpha),
     reinterpret_cast<const complex32*>(x), &incx,
     reinterpret_cast<complex32*>(y), &incy);
}
template <>
typename blas_funcs<scalars::float_complex, float>::scalar
blas_funcs<scalars::float_complex, float>::dot(
        const integer n, const scalar* x, const integer incx, const scalar* y,
        const integer incy
) noexcept
{
    scalar result;
    RPY_BLAS_FUNC(cdotc)
    (reinterpret_cast<complex32*>(&result), &n,
     reinterpret_cast<const complex32*>(x), &incx,
     reinterpret_cast<const complex32*>(y), &incy);
    return result;
}
template <>
typename blas_funcs<scalars::float_complex, float>::abs_scalar
blas_funcs<scalars::float_complex, float>::asum(
        const integer n, const scalar* x, const integer incx
) noexcept
{
    return RPY_BLAS_FUNC(scasum)(
            &n, reinterpret_cast<const complex32*>(x), &incx
    );
}
template <>
typename blas_funcs<scalars::float_complex, float>::abs_scalar
blas_funcs<scalars::float_complex, float>::nrm2(
        const integer n, const scalar* x, const integer incx
) noexcept
{
    return RPY_BLAS_FUNC(scnrm2)(
            &n, reinterpret_cast<const complex32*>(x), &incx
    );
}
template <>
integer blas_funcs<scalars::float_complex, float>::iamax(
        const integer n, const scalar* x, const integer incx
) noexcept
{
    return RPY_BLAS_FUNC(icamax)(
            &n, reinterpret_cast<const complex32*>(x), &incx
    );
}
template <>
void blas_funcs<scalars::float_complex, float>::gemv(
        BlasTranspose trans, const integer m, const integer n,
        const scalar& alpha, const scalar* A, const integer lda,
        const scalar* x, const integer incx, const scalar& beta, scalar* y,
        const integer incy
) noexcept
{
    RPY_BLAS_FUNC(cgemv)
    (reinterpret_cast<const char*>(&trans), &m, &n,
     reinterpret_cast<const complex32*>(&alpha),
     reinterpret_cast<const complex32*>(A), &lda,
     reinterpret_cast<const complex32*>(x), &incx,
     reinterpret_cast<const complex32*>(&beta), reinterpret_cast<complex32*>(y),
     &incy);
}
template <>
void blas_funcs<scalars::float_complex, float>::gemm(
        BlasTranspose transa, BlasTranspose transb, const integer m,
        const integer n, const integer k, const scalar& alpha, const scalar* A,
        const integer lda, const scalar* B, const integer ldb,
        const scalar& beta, scalar* C, const integer ldc
) noexcept
{
    RPY_BLAS_FUNC(cgemm)
    (reinterpret_cast<const char*>(&transa),
     reinterpret_cast<const char*>(&transb), &m, &n, &k,
     reinterpret_cast<const complex32*>(&alpha),
     reinterpret_cast<const complex32*>(A), &lda,
     reinterpret_cast<const complex32*>(B), &ldb,
     reinterpret_cast<const complex32*>(&beta), reinterpret_cast<complex32*>(C),
     &ldc);
}

template <>
void blas_funcs<scalars::double_complex, double>::axpy(
        const integer n, const scalar& alpha, const scalar* x,
        const integer incx, scalar* y, const integer incy
) noexcept
{
    RPY_BLAS_FUNC(zaxpy)
    (&n, reinterpret_cast<const complex64*>(&alpha),
     reinterpret_cast<const complex64*>(x), &incx,
     reinterpret_cast<complex64*>(y), &incy);
}
template <>
typename blas_funcs<scalars::double_complex, double>::scalar
blas_funcs<scalars::double_complex, double>::dot(
        const integer n, const scalar* x, const integer incx, const scalar* y,
        const integer incy
) noexcept
{
    scalar result;
    RPY_BLAS_FUNC(zdotc)
    (reinterpret_cast<complex64*>(&result), &n,
     reinterpret_cast<const complex64*>(x), &incx,
     reinterpret_cast<const complex64*>(y), &incy);
    return result;
}
template <>
typename blas_funcs<scalars::double_complex, double>::abs_scalar
blas_funcs<scalars::double_complex, double>::asum(
        const integer n, const scalar* x, const integer incx
) noexcept
{
    return RPY_BLAS_FUNC(dzasum)(
            &n, reinterpret_cast<const complex64*>(x), &incx
    );
}
template <>
typename blas_funcs<scalars::double_complex, double>::abs_scalar
blas_funcs<scalars::double_complex, double>::nrm2(
        const integer n, const scalar* x, const integer incx
) noexcept
{
    return RPY_BLAS_FUNC(dznrm2)(
            &n, reinterpret_cast<const complex64*>(x), &incx
    );
}
template <>
integer blas_funcs<scalars::double_complex, double>::iamax(
        const integer n, const scalar* x, const integer incx
) noexcept
{
    return RPY_BLAS_FUNC(izamax)(
            &n, reinterpret_cast<const complex64*>(x), &incx
    );
}
template <>
void blas_funcs<scalars::double_complex, double>::gemv(
        BlasTranspose trans, const integer m, const integer n,
        const scalar& alpha, const scalar* A, const integer lda,
        const scalar* x, const integer incx, const scalar& beta, scalar* y,
        const integer incy
) noexcept
{
    RPY_BLAS_FUNC(zgemv)
    (reinterpret_cast<const char*>(&trans), &m, &n,
     reinterpret_cast<const complex64*>(&alpha),
     reinterpret_cast<const complex64*>(A), &lda,
     reinterpret_cast<const complex64*>(x), &incx,
     reinterpret_cast<const complex64*>(&beta), reinterpret_cast<complex64*>(y),
     &incy);
}
template <>
void blas_funcs<scalars::double_complex, double>::gemm(
        BlasTranspose transa, BlasTranspose transb, const integer m,
        const integer n, const integer k, const scalar& alpha, const scalar* A,
        const integer lda, const scalar* B, const integer ldb,
        const scalar& beta, scalar* C, const integer ldc
) noexcept
{
    RPY_BLAS_FUNC(zgemm)
    (reinterpret_cast<const char*>(&transa),
     reinterpret_cast<const char*>(&transb), &m, &n, &k,
     reinterpret_cast<const complex64*>(&alpha),
     reinterpret_cast<const complex64*>(A), &lda,
     reinterpret_cast<const complex64*>(B), &ldb,
     reinterpret_cast<const complex64*>(&beta), reinterpret_cast<complex64*>(C),
     &ldc);
}

}// namespace blas

namespace lapack {

template <>
struct lapack_func_workspace<float, float> {
    std::vector<float> m_workspace;
    std::vector<integer> m_iwork;
    std::vector<integer> m_lwork;
    std::vector<float> m_rwork;
};

template <>
struct lapack_func_workspace<double, double> {
    std::vector<double> m_workspace;
    std::vector<integer> m_iwork;
    std::vector<integer> m_lwork;
    std::vector<double> m_rwork;
};

template <>
struct lapack_func_workspace<scalars::float_complex, float> {
    std::vector<scalars::float_complex> m_workspace;
    std::vector<integer> m_iwork;
    std::vector<integer> m_lwork;
    std::vector<float> m_rwork;
};

template <>
struct lapack_func_workspace<scalars::double_complex, double> {
    std::vector<scalars::double_complex> m_workspace;
    std::vector<integer> m_iwork;
    std::vector<integer> m_lwork;
    std::vector<double> m_rwork;
};

template <>
void lapack_funcs<float, float>::gesv(
        const integer n, const integer nrhs, float* A, const integer lda,
        integer* ipiv, float* B, const integer ldb, integer& info
) noexcept
{
    RPY_LAPACK_FUNC(sgesv)(&n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
}
template <>
void lapack_funcs<float, float>::syev(
        const char* jobz, blas::BlasUpLo uplo, const integer n, float* A,
        const integer lda, float* w, float* work, const integer* lwork,
        float* rwork, integer& info
) noexcept
{
    RPY_LAPACK_FUNC(ssyev)
    (jobz, reinterpret_cast<const char*>(&uplo), &n, A, &lda, w, work, lwork,
     &info);
}
template <>
void lapack_funcs<float, float>::geev(
        const char* joblv, const char* jobvr, const integer n, float* A,
        const integer lda, float* wr, float* wi, float* vl, const integer ldvl,
        float* vr, const integer ldvr, float* work, const integer* lwork,
        float* rwork, integer& info
) noexcept
{
    RPY_LAPACK_FUNC(sgeev)
    (joblv, jobvr, &n, A, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, lwork,
     &info);
}
template <>
void lapack_funcs<float, float>::gesvd(
        const char* jobu, const char* jobvt, const integer m, const integer n,
        float* A, const integer lda, float* s, float* u, const integer ldu,
        float* vt, const integer ldvt, float* work, const integer* lwork,
        float* rwork, integer& info
) noexcept
{
    RPY_LAPACK_FUNC(sgesvd)
    (jobu, jobvt, &m, &n, A, &lda, s, u, &ldu, vt, &ldvt, work, lwork, &info);
}
template <>
void lapack_funcs<float, float>::gesdd(
        const char* jobz, const integer m, const integer n, float* A,
        const integer lda, float* s, float* u, const integer ldu, float* vt,
        const integer ldvt, float* work, const integer* lwork, integer* iwork,
        float* rwork, integer& info
) noexcept
{
    RPY_LAPACK_FUNC(sgesdd)
    (jobz, &m, &n, A, &lda, s, u, &ldu, vt, &ldvt, work, lwork, iwork, &info);
}
template <>
void lapack_funcs<float, float>::gels(
        blas::BlasTranspose trans, const integer m, const integer n,
        const integer nrhs, float* A, const integer lda, float* B,
        const integer ldb, float* work, const integer* lwork, integer& info
) noexcept
{
    RPY_LAPACK_FUNC(sgels)
    (reinterpret_cast<const char*>(trans), &m, &n, &nrhs, A, &lda, B, &ldb,
     work, lwork, &info);
}
template <>
void lapack_funcs<float, float>::gelsy(
        const integer m, const integer n, const integer nrhs, float* A,
        const integer lda, float* B, const integer ldb, integer* jpvt,
        const float& rcond, integer& rank, float* work, const integer* lwork,
        float* rwork, integer& info
) noexcept
{
    RPY_LAPACK_FUNC(sgelsy)
    (&m, &n, &nrhs, A, &lda, B, &ldb, jpvt, &rcond, &rank, work, lwork, &info);
}
template <>
void lapack_funcs<float, float>::gelss(
        const integer m, const integer n, const integer nrhs, float* A,
        const integer lda, float* B, const integer ldb, float* s,
        const float& rcond, integer& rank, float* work, const integer* lwork,
        float* rwork, integer& info
) noexcept
{
    RPY_LAPACK_FUNC(sgelss)
    (&m, &n, &nrhs, A, &lda, B, &ldb, s, &rcond, &rank, work, lwork, &info);
}
template <>
void lapack_funcs<float, float>::gelsd(
        const integer m, const integer n, const integer nrhs, float* A,
        const integer lda, float* B, const integer ldb, float* s,
        const float& rcond, integer& rank, float* work, const integer* lwork,
        float* rwork, integer* iwork, integer& info
) noexcept
{
    RPY_LAPACK_FUNC(sgelsd)
    (&m, &n, &nrhs, A, &lda, B, &ldb, s, &rcond, &rank, work, lwork, iwork,
     &info);
}

template <>
void lapack_funcs<double, double>::gesv(
        const integer n, const integer nrhs, double* A, const integer lda,
        integer* ipiv, double* B, const integer ldb, integer& info
) noexcept
{
    RPY_LAPACK_FUNC(dgesv)(&n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
}
template <>
void lapack_funcs<double, double>::syev(
        const char* jobz, blas::BlasUpLo uplo, const integer n, double* A,
        const integer lda, double* w, double* work, const integer* lwork,
        double* rwork, integer& info
) noexcept
{
    RPY_LAPACK_FUNC(dsyev)
    (jobz, reinterpret_cast<const char*>(&uplo), &n, A, &lda, w, work, lwork,
     &info);
}
template <>
void lapack_funcs<double, double>::geev(
        const char* joblv, const char* jobvr, const integer n, double* A,
        const integer lda, double* wr, double* wi, double* vl,
        const integer ldvl, double* vr, const integer ldvr, double* work,
        const integer* lwork, double* rwork, integer& info
) noexcept
{
    RPY_LAPACK_FUNC(dgeev)
    (joblv, jobvr, &n, A, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, lwork,
     &info);
}
template <>
void lapack_funcs<double, double>::gesvd(
        const char* jobu, const char* jobvt, const integer m, const integer n,
        double* A, const integer lda, double* s, double* u, const integer ldu,
        double* vt, const integer ldvt, double* work, const integer* lwork,
        double* rwork, integer& info
) noexcept
{
    RPY_LAPACK_FUNC(dgesvd)
    (jobu, jobvt, &m, &n, A, &lda, s, u, &ldu, vt, &ldvt, work, lwork, &info);
}
template <>
void lapack_funcs<double, double>::gesdd(
        const char* jobz, const integer m, const integer n, double* A,
        const integer lda, double* s, double* u, const integer ldu, double* vt,
        const integer ldvt, double* work, const integer* lwork, integer* iwork,
        double* rwork, integer& info
) noexcept
{
    RPY_LAPACK_FUNC(dgesdd)
    (jobz, &m, &n, A, &lda, s, u, &ldu, vt, &ldvt, work, lwork, iwork, &info);
}
template <>
void lapack_funcs<double, double>::gels(
        blas::BlasTranspose trans, const integer m, const integer n,
        const integer nrhs, double* A, const integer lda, double* B,
        const integer ldb, double* work, const integer* lwork, integer& info
) noexcept
{
    RPY_LAPACK_FUNC(dgels)
    (reinterpret_cast<const char*>(trans), &m, &n, &nrhs, A, &lda, B, &ldb,
     work, lwork, &info);
}
template <>
void lapack_funcs<double, double>::gelsy(
        const integer m, const integer n, const integer nrhs, double* A,
        const integer lda, double* B, const integer ldb, integer* jpvt,
        const double& rcond, integer& rank, double* work, const integer* lwork,
        double* rwork, integer& info
) noexcept
{
    RPY_LAPACK_FUNC(dgelsy)
    (&m, &n, &nrhs, A, &lda, B, &ldb, jpvt, &rcond, &rank, work, lwork, &info);
}
template <>
void lapack_funcs<double, double>::gelss(
        const integer m, const integer n, const integer nrhs, double* A,
        const integer lda, double* B, const integer ldb, double* s,
        const double& rcond, integer& rank, double* work, const integer* lwork,
        double* rwork, integer& info
) noexcept
{
    RPY_LAPACK_FUNC(dgelss)
    (&m, &n, &nrhs, A, &lda, B, &ldb, s, &rcond, &rank, work, lwork, &info);
}
template <>
void lapack_funcs<double, double>::gelsd(
        const integer m, const integer n, const integer nrhs, double* A,
        const integer lda, double* B, const integer ldb, double* s,
        const double& rcond, integer& rank, double* work, const integer* lwork,
        double* rwork, integer* iwork, integer& info
) noexcept
{
    RPY_LAPACK_FUNC(dgelsd)
    (&m, &n, &nrhs, A, &lda, B, &ldb, s, &rcond, &rank, work, lwork, iwork,
     &info);
}

template <>
void lapack_funcs<scalars::float_complex, float>::gesv(
        const integer n, const integer nrhs, scalars::float_complex* A,
        const integer lda, integer* ipiv, scalars::float_complex* B,
        const integer ldb, integer& info
) noexcept
{
    RPY_LAPACK_FUNC(cgesv)
    (&n, &nrhs, reinterpret_cast<complex32*>(A), &lda, ipiv,
     reinterpret_cast<complex32*>(B), &ldb, &info);
}
template <>
void lapack_funcs<scalars::float_complex, float>::syev(
        const char* jobz, blas::BlasUpLo uplo, const integer n,
        scalars::float_complex* A, const integer lda, float* w,
        scalars::float_complex* work, const integer* lwork, float* rwork,
        integer& info
) noexcept
{
    RPY_LAPACK_FUNC(cheev)
    (jobz, reinterpret_cast<const char*>(&uplo), &n,
     reinterpret_cast<complex32*>(A), &lda, w,
     reinterpret_cast<complex32*>(work), lwork, rwork, &info);
}
template <>
void lapack_funcs<scalars::float_complex, float>::geev(
        const char* joblv, const char* jobvr, const integer n,
        scalars::float_complex* A, const integer lda,
        scalars::float_complex* wr, scalars::float_complex* wi,
        scalars::float_complex* vl, const integer ldvl,
        scalars::float_complex* vr, const integer ldvr,
        scalars::float_complex* work, const integer* lwork, float* rwork,
        integer& info
) noexcept
{
    RPY_LAPACK_FUNC(cgeev)
    (joblv, jobvr, &n, reinterpret_cast<complex32*>(A), &lda,
     reinterpret_cast<complex32*>(wr), reinterpret_cast<complex32*>(vl), &ldvl,
     reinterpret_cast<complex32*>(vr), &ldvr,
     reinterpret_cast<complex32*>(work), lwork, rwork, &info);
}
template <>
void lapack_funcs<scalars::float_complex, float>::gesvd(
        const char* jobu, const char* jobvt, const integer m, const integer n,
        scalars::float_complex* A, const integer lda, float* s,
        scalars::float_complex* u, const integer ldu,
        scalars::float_complex* vt, const integer ldvt,
        scalars::float_complex* work, const integer* lwork, float* rwork,
        integer& info
) noexcept
{
    RPY_LAPACK_FUNC(cgesvd)
    (jobu, jobvt, &m, &n, reinterpret_cast<complex32*>(A), &lda, s,
     reinterpret_cast<complex32*>(u), &ldu, reinterpret_cast<complex32*>(vt),
     &ldvt, reinterpret_cast<complex32*>(work), lwork, rwork, &info);
}
template <>
void lapack_funcs<scalars::float_complex, float>::gesdd(
        const char* jobz, const integer m, const integer n,
        scalars::float_complex* A, const integer lda, float* s,
        scalars::float_complex* u, const integer ldu,
        scalars::float_complex* vt, const integer ldvt,
        scalars::float_complex* work, const integer* lwork, integer* iwork,
        float* rwork, integer& info
) noexcept
{
    RPY_LAPACK_FUNC(cgesdd)
    (jobz, &m, &n, reinterpret_cast<complex32*>(A), &lda, s,
     reinterpret_cast<complex32*>(u), &ldu, reinterpret_cast<complex32*>(vt),
     &ldvt, reinterpret_cast<complex32*>(work), lwork, rwork, iwork, &info);
}
template <>
void lapack_funcs<scalars::float_complex, float>::gels(
        blas::BlasTranspose trans, const integer m, const integer n,
        const integer nrhs, scalars::float_complex* A, const integer lda,
        scalars::float_complex* B, const integer ldb,
        scalars::float_complex* work, const integer* lwork, integer& info
) noexcept
{
    RPY_LAPACK_FUNC(cgels)
    (reinterpret_cast<const char*>(trans), &m, &n, &nrhs,
     reinterpret_cast<complex32*>(A), &lda, reinterpret_cast<complex32*>(B),
     &ldb, reinterpret_cast<complex32*>(work), lwork, &info);
}
template <>
void lapack_funcs<scalars::float_complex, float>::gelsy(
        const integer m, const integer n, const integer nrhs,
        scalars::float_complex* A, const integer lda, scalars::float_complex* B,
        const integer ldb, integer* jpvt, const float& rcond, integer& rank,
        scalars::float_complex* work, const integer* lwork, float* rwork,
        integer& info
) noexcept
{
    RPY_LAPACK_FUNC(cgelsy)
    (&m, &n, &nrhs, reinterpret_cast<complex32*>(A), &lda,
     reinterpret_cast<complex32*>(B), &ldb, jpvt, &rcond, &rank,
     reinterpret_cast<complex32*>(work), lwork, rwork, &info);
}
template <>
void lapack_funcs<scalars::float_complex, float>::gelss(
        const integer m, const integer n, const integer nrhs,
        scalars::float_complex* A, const integer lda, scalars::float_complex* B,
        const integer ldb, float* s, const float& rcond, integer& rank,
        scalars::float_complex* work, const integer* lwork, float* rwork,
        integer& info
) noexcept
{
    RPY_LAPACK_FUNC(cgelss)
    (&m, &n, &nrhs, reinterpret_cast<complex32*>(A), &lda,
     reinterpret_cast<complex32*>(B), &ldb, s, &rcond, &rank,
     reinterpret_cast<complex32*>(work), lwork, rwork, &info);
}
template <>
void lapack_funcs<scalars::float_complex, float>::gelsd(
        const integer m, const integer n, const integer nrhs,
        scalars::float_complex* A, const integer lda, scalars::float_complex* B,
        const integer ldb, float* s, const float& rcond, integer& rank,
        scalars::float_complex* work, const integer* lwork, float* rwork,
        integer* iwork, integer& info
) noexcept
{
    RPY_LAPACK_FUNC(cgelsd)
    (&m, &n, &nrhs, reinterpret_cast<complex32*>(A), &lda,
     reinterpret_cast<complex32*>(B), &ldb, s, &rcond, &rank,
     reinterpret_cast<complex32*>(work), lwork, rwork, iwork, &info);
}

template <>
void lapack_funcs<scalars::double_complex, double>::gesv(
        const integer n, const integer nrhs, scalars::double_complex* A,
        const integer lda, integer* ipiv, scalars::double_complex* B,
        const integer ldb, integer& info
) noexcept
{
    RPY_LAPACK_FUNC(zgesv)
    (&n, &nrhs, reinterpret_cast<complex64*>(A), &lda, ipiv,
     reinterpret_cast<complex64*>(B), &ldb, &info);
}
template <>
void lapack_funcs<scalars::double_complex, double>::syev(
        const char* jobz, blas::BlasUpLo uplo, const integer n,
        scalars::double_complex* A, const integer lda, double* w,
        scalars::double_complex* work, const integer* lwork, double* rwork,
        integer& info
) noexcept
{
    RPY_LAPACK_FUNC(zheev)
    (jobz, reinterpret_cast<const char*>(&uplo), &n,
     reinterpret_cast<complex64*>(A), &lda, w,
     reinterpret_cast<complex64*>(work), lwork, rwork, &info);
}
template <>
void lapack_funcs<scalars::double_complex, double>::geev(
        const char* joblv, const char* jobvr, const integer n,
        scalars::double_complex* A, const integer lda,
        scalars::double_complex* wr, scalars::double_complex* wi,
        scalars::double_complex* vl, const integer ldvl,
        scalars::double_complex* vr, const integer ldvr,
        scalars::double_complex* work, const integer* lwork, double* rwork,
        integer& info
) noexcept
{
    RPY_LAPACK_FUNC(zgeev)
    (joblv, jobvr, &n, reinterpret_cast<complex64*>(A), &lda,
     reinterpret_cast<complex64*>(wr), reinterpret_cast<complex64*>(vl), &ldvl,
     reinterpret_cast<complex64*>(vr), &ldvr,
     reinterpret_cast<complex64*>(work), lwork, rwork, &info);
}
template <>
void lapack_funcs<scalars::double_complex, double>::gesvd(
        const char* jobu, const char* jobvt, const integer m, const integer n,
        scalars::double_complex* A, const integer lda, double* s,
        scalars::double_complex* u, const integer ldu,
        scalars::double_complex* vt, const integer ldvt,
        scalars::double_complex* work, const integer* lwork, double* rwork,
        integer& info
) noexcept
{
    RPY_LAPACK_FUNC(zgesvd)
    (jobu, jobvt, &m, &n, reinterpret_cast<complex64*>(A), &lda, s,
     reinterpret_cast<complex64*>(u), &ldu, reinterpret_cast<complex64*>(vt),
     &ldvt, reinterpret_cast<complex64*>(work), lwork, rwork, &info);
}
template <>
void lapack_funcs<scalars::double_complex, double>::gesdd(
        const char* jobz, const integer m, const integer n,
        scalars::double_complex* A, const integer lda, double* s,
        scalars::double_complex* u, const integer ldu,
        scalars::double_complex* vt, const integer ldvt,
        scalars::double_complex* work, const integer* lwork, integer* iwork,
        double* rwork, integer& info
) noexcept
{
    RPY_LAPACK_FUNC(zgesdd)
    (jobz, &m, &n, reinterpret_cast<complex64*>(A), &lda, s,
     reinterpret_cast<complex64*>(u), &ldu, reinterpret_cast<complex64*>(vt),
     &ldvt, reinterpret_cast<complex64*>(work), lwork, rwork, iwork, &info);
}
template <>
void lapack_funcs<scalars::double_complex, double>::gels(
        blas::BlasTranspose trans, const integer m, const integer n,
        const integer nrhs, scalars::double_complex* A, const integer lda,
        scalars::double_complex* B, const integer ldb,
        scalars::double_complex* work, const integer* lwork, integer& info
) noexcept
{
    RPY_LAPACK_FUNC(zgels)
    (reinterpret_cast<const char*>(trans), &m, &n, &nrhs,
     reinterpret_cast<complex64*>(A), &lda, reinterpret_cast<complex64*>(B),
     &ldb, reinterpret_cast<complex64*>(work), lwork, &info);
}
template <>
void lapack_funcs<scalars::double_complex, double>::gelsy(
        const integer m, const integer n, const integer nrhs,
        scalars::double_complex* A, const integer lda,
        scalars::double_complex* B, const integer ldb, integer* jpvt,
        const double& rcond, integer& rank, scalars::double_complex* work,
        const integer* lwork, double* rwork, integer& info
) noexcept
{
    RPY_LAPACK_FUNC(zgelsy)
    (&m, &n, &nrhs, reinterpret_cast<complex64*>(A), &lda,
     reinterpret_cast<complex64*>(B), &ldb, jpvt, &rcond, &rank,
     reinterpret_cast<complex64*>(work), lwork, rwork, &info);
}
template <>
void lapack_funcs<scalars::double_complex, double>::gelss(
        const integer m, const integer n, const integer nrhs,
        scalars::double_complex* A, const integer lda,
        scalars::double_complex* B, const integer ldb, double* s,
        const double& rcond, integer& rank, scalars::double_complex* work,
        const integer* lwork, double* rwork, integer& info
) noexcept
{
    RPY_LAPACK_FUNC(zgelss)
    (&m, &n, &nrhs, reinterpret_cast<complex64*>(A), &lda,
     reinterpret_cast<complex64*>(B), &ldb, s, &rcond, &rank,
     reinterpret_cast<complex64*>(work), lwork, rwork, &info);
}
template <>
void lapack_funcs<scalars::double_complex, double>::gelsd(
        const integer m, const integer n, const integer nrhs,
        scalars::double_complex* A, const integer lda,
        scalars::double_complex* B, const integer ldb, double* s,
        const double& rcond, integer& rank, scalars::double_complex* work,
        const integer* lwork, double* rwork, integer* iwork, integer& info
) noexcept
{
    RPY_LAPACK_FUNC(zgelsd)
    (&m, &n, &nrhs, reinterpret_cast<complex64*>(A), &lda,
     reinterpret_cast<complex64*>(B), &ldb, s, &rcond, &rank,
     reinterpret_cast<complex64*>(work), lwork, rwork, iwork, &info);
}

}// namespace lapack

}// namespace rpy
