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

#include "scalar_blas_defs.h"
#include <roughpy/scalars/scalars_fwd.h>

#define RPY_BLAS_FUNC(NAME) NAME
#define RPY_LAPACK_FUNC(NAME) NAME

namespace rpy {
namespace blas {

template <typename S, typename R>
struct blas_funcs;

template <>
struct blas_funcs<float, float> {
    using scalar = float;
    using abs_scalar = float;

    // Level 1 functions
    inline static void
    axpy(const integer n, const scalar& alpha, const scalar* RPY_RESTRICT x,
         const integer incx, scalar* RPY_RESTRICT y,
         const integer incy) noexcept
    {
        RPY_BLAS_FUNC(saxpy)(&n, &alpha, x, &incx, y, &incy);
    }

    inline static scalar
    dot(const integer n, const scalar* RPY_RESTRICT x, const integer incx,
        const scalar* RPY_RESTRICT y, const integer incy) noexcept
    {
        return RPY_BLAS_FUNC(sdot)(&n, x, &incx, y, &incy);
    }

    inline static abs_scalar
    asum(const integer n, const scalar* RPY_RESTRICT x,
         const integer incx) noexcept
    {
        return RPY_BLAS_FUNC(sasum)(&n, x, &incx);
    }

    inline static abs_scalar
    nrm2(const integer n, const scalar* RPY_RESTRICT x,
         const integer incx) noexcept
    {
        return RPY_BLAS_FUNC(snrm2)(&n, x, &incx);
    }

    inline static integer
    iamax(const integer n, const scalar* RPY_RESTRICT x,
          const integer incx) noexcept
    {
        return RPY_BLAS_FUNC(isamax)(&n, x, &incx);
    }

    // Level 2

    inline static void
    gemv(BlasTranspose trans, const integer m, const integer n,
         const scalar& alpha, const scalar* RPY_RESTRICT A, const integer lda,
         const scalar* RPY_RESTRICT x, const integer incx, const scalar& beta,
         scalar* RPY_RESTRICT y, const integer incy) noexcept
    {
        RPY_BLAS_FUNC(sgemv)
        (reinterpret_cast<const char*>(&trans), &m, &n, &alpha, A, &lda, x,
         &incx, &beta, y, &incy);
    }

    // Level 3

    inline static void
    gemm(BlasTranspose transa, BlasTranspose transb, const integer m,
         const integer n, const integer k, const scalar& alpha,
         const scalar* RPY_RESTRICT A, const integer lda,
         const scalar* RPY_RESTRICT b, const integer ldb, const scalar& beta,
         scalar* RPY_RESTRICT C, const integer ldc)
    {
        RPY_BLAS_FUNC(sgemm)
        (reinterpret_cast<const char*>(&transa),
         reinterpret_cast<const char*>(&transb), &m, &n, &k, &alpha, A, &lda, b,
         &ldb, &beta, C, &ldc);
    }
};

template <>
struct blas_funcs<double, double> {
    using scalar = double;
    using abs_scalar = double;

    // Level 1 functions
    inline static void
    axpy(const integer n, const scalar& alpha, const scalar* RPY_RESTRICT x,
         const integer incx, scalar* RPY_RESTRICT y,
         const integer incy) noexcept
    {
        RPY_BLAS_FUNC(daxpy)(&n, &alpha, x, &incx, y, &incy);
    }

    inline static scalar
    dot(const integer n, const scalar* RPY_RESTRICT x, const integer incx,
        const scalar* RPY_RESTRICT y, const integer incy) noexcept
    {
        return RPY_BLAS_FUNC(ddot)(&n, x, &incx, y, &incy);
    }

    inline static abs_scalar
    asum(const integer n, const scalar* RPY_RESTRICT x,
         const integer incx) noexcept
    {
        return RPY_BLAS_FUNC(dasum)(&n, x, &incx);
    }

    inline static abs_scalar
    nrm2(const integer n, const scalar* RPY_RESTRICT x,
         const integer incx) noexcept
    {
        return RPY_BLAS_FUNC(dnrm2)(&n, x, &incx);
    }

    inline static integer
    iamax(const integer n, const scalar* RPY_RESTRICT x,
          const integer incx) noexcept
    {
        return RPY_BLAS_FUNC(idamax)(&n, x, &incx);
    }

    // Level 2

    inline static void
    gemv(BlasTranspose trans, const integer m, const integer n,
         const scalar& alpha, const scalar* RPY_RESTRICT A, const integer lda,
         const scalar* RPY_RESTRICT x, const integer incx, const scalar& beta,
         scalar* RPY_RESTRICT y, const integer incy) noexcept
    {
        RPY_BLAS_FUNC(dgemv)
        (reinterpret_cast<const char*>(&trans), &m, &n, &alpha, A, &lda, x,
         &incx, &beta, y, &incy);
    }

    // Level 3

    inline static void
    gemm(BlasTranspose transa, BlasTranspose transb, const integer m,
         const integer n, const integer k, const scalar& alpha,
         const scalar* RPY_RESTRICT A, const integer lda,
         const scalar* RPY_RESTRICT b, const integer ldb, const scalar& beta,
         scalar* RPY_RESTRICT C, const integer ldc)
    {
        RPY_BLAS_FUNC(dgemm)
        (reinterpret_cast<const char*>(&transa),
         reinterpret_cast<const char*>(&transb), &m, &n, &k, &alpha, A, &lda, b,
         &ldb, &beta, C, &ldc);
    }
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
    {
        RPY_BLAS_FUNC(caxpy)
        (&n, reinterpret_cast<const complex32*>(&alpha),
         reinterpret_cast<const complex32*>(x), &incx,
         reinterpret_cast<complex32*>(y), &incy);
    }

    inline static scalar
    dot(const integer n, const scalar* RPY_RESTRICT x, const integer incx,
        const scalar* RPY_RESTRICT y, const integer incy) noexcept
    {
        scalar result;
        RPY_BLAS_FUNC(cdotc)
        (reinterpret_cast<complex32*>(&result), &n,
         reinterpret_cast<const complex32*>(x), &incx,
         reinterpret_cast<const complex32*>(y), &incy);
        return result;
    }

    inline static abs_scalar
    asum(const integer n, const scalar* RPY_RESTRICT x,
         const integer incx) noexcept
    {
        return RPY_BLAS_FUNC(scasum)(
                &n, reinterpret_cast<const complex32*>(x), &incx
        );
    }

    inline static abs_scalar
    nrm2(const integer n, const scalar* RPY_RESTRICT x,
         const integer incx) noexcept
    {
        return RPY_BLAS_FUNC(scnrm2)(
                &n, reinterpret_cast<const complex32*>(x), &incx
        );
    }

    inline static integer
    iamax(const integer n, const scalar* RPY_RESTRICT x,
          const integer incx) noexcept
    {
        return RPY_BLAS_FUNC(icamax)(
                &n, reinterpret_cast<const complex32*>(x), &incx
        );
    }

    // Level 2

    inline static void
    gemv(BlasTranspose trans, const integer m, const integer n,
         const scalar& alpha, const scalar* RPY_RESTRICT A, const integer lda,
         const scalar* RPY_RESTRICT x, const integer incx, const scalar& beta,
         scalar* RPY_RESTRICT y, const integer incy) noexcept
    {
        RPY_BLAS_FUNC(cgemv)
        (reinterpret_cast<const char*>(&trans), &m, &n,
         reinterpret_cast<const complex32*>(&alpha),
         reinterpret_cast<const complex32*>(A), &lda,
         reinterpret_cast<const complex32*>(x), &incx,
         reinterpret_cast<const complex32*>(&beta),
         reinterpret_cast<complex32*>(y), &incy);
    }

    // Level 3

    inline static void
    gemm(BlasTranspose transa, BlasTranspose transb, const integer m,
         const integer n, const integer k, const scalar& alpha,
         const scalar* RPY_RESTRICT A, const integer lda,
         const scalar* RPY_RESTRICT B, const integer ldb, const scalar& beta,
         scalar* RPY_RESTRICT C, const integer ldc)
    {
        RPY_BLAS_FUNC(cgemm)
        (reinterpret_cast<const char*>(&transa),
         reinterpret_cast<const char*>(&transb), &m, &n, &k,
         reinterpret_cast<const complex32*>(&alpha),
         reinterpret_cast<const complex32*>(A), &lda,
         reinterpret_cast<const complex32*>(B), &ldb,
         reinterpret_cast<const complex32*>(&beta),
         reinterpret_cast<complex32*>(C), &ldc);
    }
};

template <>
struct blas_funcs<scalars::double_complex, double> {
    using scalar = scalars::double_complex;
    using abs_scalar = double;

    // Level 1 functions
    inline static void
    axpy(const integer n, const scalar& alpha, const scalar* RPY_RESTRICT x,
         const integer incx, scalar* RPY_RESTRICT y,
         const integer incy) noexcept
    {
        RPY_BLAS_FUNC(zaxpy)
        (&n, reinterpret_cast<const complex64*>(&alpha),
         reinterpret_cast<const complex64*>(x), &incx,
         reinterpret_cast<complex64*>(y), &incy);
    }

    inline static scalar
    dot(const integer n, const scalar* RPY_RESTRICT x, const integer incx,
        const scalar* RPY_RESTRICT y, const integer incy) noexcept
    {
        scalar result;
        RPY_BLAS_FUNC(zdotc)
        (reinterpret_cast<complex64*>(&result), &n,
         reinterpret_cast<const complex64*>(x), &incx,
         reinterpret_cast<const complex64*>(y), &incy);
        return result;
    }

    inline static abs_scalar
    asum(const integer n, const scalar* RPY_RESTRICT x,
         const integer incx) noexcept
    {
        return RPY_BLAS_FUNC(dzasum)(
                &n, reinterpret_cast<const complex64*>(x), &incx
        );
    }

    inline static abs_scalar
    nrm2(const integer n, const scalar* RPY_RESTRICT x,
         const integer incx) noexcept
    {
        return RPY_BLAS_FUNC(dznrm2)(
                &n, reinterpret_cast<const complex64*>(x), &incx
        );
    }

    inline static integer
    iamax(const integer n, const scalar* RPY_RESTRICT x,
          const integer incx) noexcept
    {
        return RPY_BLAS_FUNC(izamax)(
                &n, reinterpret_cast<const complex64*>(x), &incx
        );
    }

    // Level 2

    inline static void
    gemv(BlasTranspose trans, const integer m, const integer n,
         const scalar& alpha, const scalar* RPY_RESTRICT A, const integer lda,
         const scalar* RPY_RESTRICT x, const integer incx, const scalar& beta,
         scalar* RPY_RESTRICT y, const integer incy) noexcept
    {
        RPY_BLAS_FUNC(zgemv)
        (reinterpret_cast<const char*>(&trans), &m, &n,
         reinterpret_cast<const complex64*>(&alpha),
         reinterpret_cast<const complex64*>(A), &lda,
         reinterpret_cast<const complex64*>(x), &incx,
         reinterpret_cast<const complex64*>(&beta),
         reinterpret_cast<complex64*>(y), &incy);
    }

    // Level 3

    inline static void
    gemm(BlasTranspose transa, BlasTranspose transb, const integer m,
         const integer n, const integer k, const scalar& alpha,
         const scalar* RPY_RESTRICT A, const integer lda,
         const scalar* RPY_RESTRICT B, const integer ldb, const scalar& beta,
         scalar* RPY_RESTRICT C, const integer ldc)
    {
        RPY_BLAS_FUNC(zgemm)
        (reinterpret_cast<const char*>(&transa),
         reinterpret_cast<const char*>(&transb), &m, &n, &k,
         reinterpret_cast<const complex64*>(&alpha),
         reinterpret_cast<const complex64*>(A), &lda,
         reinterpret_cast<const complex64*>(B), &ldb,
         reinterpret_cast<const complex64*>(&beta),
         reinterpret_cast<complex64*>(C), &ldc);
    }
};

}// namespace blas

namespace lapack {
using blas::complex32;
using blas::complex64;
using blas::integer;
using blas::logical;

template <typename S, typename R>
struct lapack_funcs {
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

template <>
struct lapack_funcs<float, float> {
    using scalar = float;
    using real_scalar = float;

    static inline void
    gesv(const integer n, const integer nrhs, scalar* RPY_RESTRICT A,
         const integer lda, integer* RPY_RESTRICT ipiv, scalar* RPY_RESTRICT B,
         const integer ldb, integer& info) noexcept
    {
        RPY_LAPACK_FUNC(sgesv)(&n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
    }

    static inline void
    syev(const char* jobz, blas::BlasUpLo uplo, const integer n,
         scalar* RPY_RESTRICT A, const integer lda, real_scalar* RPY_RESTRICT w,
         scalar* RPY_RESTRICT work, const integer* RPY_RESTRICT lwork,
         real_scalar* RPY_RESTRICT RPY_UNUSED_VAR rwork, integer& info) noexcept
    {
        RPY_LAPACK_FUNC(ssyev)
        (jobz, reinterpret_cast<const char*>(&uplo), &n, A, &lda, w, work,
         lwork, &info);
    }

    static inline void
    geev(const char* joblv, const char* jobvr, const integer n,
         scalar* RPY_RESTRICT A, const integer lda, scalar* RPY_RESTRICT wr,
         scalar* RPY_RESTRICT wi, scalar* RPY_RESTRICT vl, const integer ldvl,
         scalar* RPY_RESTRICT vr, const integer ldvr, scalar* RPY_RESTRICT work,
         const integer* RPY_RESTRICT lwork,
         real_scalar* RPY_RESTRICT RPY_UNUSED_VAR rwork, integer& info) noexcept
    {
        RPY_LAPACK_FUNC(sgeev)
        (joblv, jobvr, &n, A, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, lwork,
         &info);
    }

    static inline void
    gesvd(const char* jobu, const char* jobvt, const integer m, const integer n,
          scalar* RPY_RESTRICT A, const integer lda, scalar* RPY_RESTRICT s,
          scalar* RPY_RESTRICT u, const integer ldu, scalar* RPY_RESTRICT vt,
          const integer ldvt, scalar* RPY_RESTRICT work,
          const integer* RPY_RESTRICT lwork,
          const integer* RPY_UNUSED_VAR rwork, integer& info) noexcept
    {
        RPY_LAPACK_FUNC(sgesvd)
        (jobu, jobvt, &m, &n, A, &lda, s, u, &ldu, vt, &ldvt, work, lwork,
         &info);
    }

    static inline void
    gesdd(const char* jobz, const integer m, const integer n,
          scalar* RPY_RESTRICT A, const integer lda, scalar* RPY_RESTRICT s,
          scalar* RPY_RESTRICT u, const integer ldu, scalar* RPY_RESTRICT vt,
          const integer ldvt, scalar* RPY_RESTRICT work,
          const integer* RPY_RESTRICT lwork, integer* RPY_RESTRICT iwork,
          const integer* RPY_UNUSED_VAR rwork, integer& info) noexcept
    {
        RPY_LAPACK_FUNC(sgesdd)
        (jobz, &m, &n, A, &lda, s, u, &ldu, vt, &ldvt, work, lwork, iwork,
         &info);
    }

    static inline void
    gels(blas::BlasTranspose trans, const integer m, const integer n,
         const integer nrhs, scalar* RPY_RESTRICT A, const integer lda,
         scalar* RPY_RESTRICT B, const integer ldb, scalar* RPY_RESTRICT work,
         const integer* RPY_RESTRICT lwork, integer& info) noexcept
    {
        RPY_LAPACK_FUNC(sgels)
        (reinterpret_cast<const char*>(trans), &m, &n, &nrhs, A, &lda, B, &ldb,
         work, lwork, &info);
    }

    static inline void
    gelsy(const integer m, const integer n, const integer nrhs,
          scalar* RPY_RESTRICT A, const integer lda, scalar* RPY_RESTRICT B,
          const integer ldb, integer* RPY_RESTRICT jpvt, const scalar& rcond,
          integer& rank, scalar* RPY_RESTRICT work,
          const integer* RPY_RESTRICT lwork,
          const integer* RPY_UNUSED_VAR rwork, integer& info) noexcept
    {
        RPY_LAPACK_FUNC(sgelsy)
        (&m, &n, &nrhs, A, &lda, B, &ldb, jpvt, &rcond, &rank, work, lwork,
         &info);
    }

    static inline void
    gelss(const integer m, const integer n, const integer nrhs,
          scalar* RPY_RESTRICT A, const integer lda, scalar* RPY_RESTRICT B,
          const integer ldb, scalar* RPY_RESTRICT s, const scalar& rcond,
          integer& rank, scalar* RPY_RESTRICT work,
          const integer* RPY_RESTRICT lwork,
          const integer* RPY_RESTRICT RPY_UNUSED_VAR rwork,
          integer& info) noexcept
    {
        RPY_LAPACK_FUNC(sgelss)
        (&m, &n, &nrhs, A, &lda, B, &ldb, s, &rcond, &rank, work, lwork, &info);
    }

    static inline void
    gelsd(const integer m, const integer n, const integer nrhs,
          scalar* RPY_RESTRICT A, const integer lda, scalar* RPY_RESTRICT B,
          const integer ldb, scalar* RPY_RESTRICT s, const scalar& rcond,
          integer& rank, scalar* RPY_RESTRICT work,
          const integer* RPY_RESTRICT lwork,
          const integer* RPY_RESTRICT RPY_UNUSED_VAR rwork,
          integer* RPY_RESTRICT iwork, integer& info) noexcept
    {
        RPY_LAPACK_FUNC(sgelsd)
        (&m, &n, &nrhs, A, &lda, B, &ldb, s, &rcond, &rank, work, lwork, iwork,
         &info);
    }
};

template <>
struct lapack_funcs<double, double> {
    using scalar = double;
    using real_scalar = double;

    static inline void
    gesv(const integer n, const integer nrhs, scalar* RPY_RESTRICT A,
         const integer lda, integer* RPY_RESTRICT ipiv, scalar* RPY_RESTRICT B,
         const integer ldb, integer& info) noexcept
    {
        RPY_LAPACK_FUNC(dgesv)(&n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
    }

    static inline void
    syev(const char* jobz, blas::BlasUpLo uplo, const integer n,
         scalar* RPY_RESTRICT A, const integer lda, real_scalar* RPY_RESTRICT w,
         scalar* RPY_RESTRICT work, const integer* RPY_RESTRICT lwork,
         real_scalar* RPY_RESTRICT RPY_UNUSED_VAR rwork, integer& info) noexcept
    {
        RPY_LAPACK_FUNC(dsyev)
        (jobz, reinterpret_cast<const char*>(&uplo), &n, A, &lda, w, work,
         lwork, &info);
    }

    static inline void
    geev(const char* joblv, const char* jobvr, const integer n,
         scalar* RPY_RESTRICT A, const integer lda, scalar* RPY_RESTRICT wr,
         scalar* RPY_RESTRICT wi, scalar* RPY_RESTRICT vl, const integer ldvl,
         scalar* RPY_RESTRICT vr, const integer ldvr, scalar* RPY_RESTRICT work,
         const integer* RPY_RESTRICT lwork,
         real_scalar* RPY_RESTRICT RPY_UNUSED_VAR rwork, integer& info) noexcept
    {
        RPY_LAPACK_FUNC(dgeev)
        (joblv, jobvr, &n, A, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, lwork,
         &info);
    }

    static inline void
    gesvd(const char* jobu, const char* jobvt, const integer m, const integer n,
          scalar* RPY_RESTRICT A, const integer lda, scalar* RPY_RESTRICT s,
          scalar* RPY_RESTRICT u, const integer ldu, scalar* RPY_RESTRICT vt,
          const integer ldvt, scalar* RPY_RESTRICT work,
          const integer* RPY_RESTRICT lwork,
          const integer* RPY_UNUSED_VAR rwork, integer& info) noexcept
    {
        RPY_LAPACK_FUNC(dgesvd)
        (jobu, jobvt, &m, &n, A, &lda, s, u, &ldu, vt, &ldvt, work, lwork,
         &info);
    }

    static inline void
    gesdd(const char* jobz, const integer m, const integer n,
          scalar* RPY_RESTRICT A, const integer lda, scalar* RPY_RESTRICT s,
          scalar* RPY_RESTRICT u, const integer ldu, scalar* RPY_RESTRICT vt,
          const integer ldvt, scalar* RPY_RESTRICT work,
          const integer* RPY_RESTRICT lwork, integer* RPY_RESTRICT iwork,
          const integer* RPY_UNUSED_VAR rwork, integer& info) noexcept
    {
        RPY_LAPACK_FUNC(dgesdd)
        (jobz, &m, &n, A, &lda, s, u, &ldu, vt, &ldvt, work, lwork, iwork,
         &info);
    }

    static inline void
    gels(blas::BlasTranspose trans, const integer m, const integer n,
         const integer nrhs, scalar* RPY_RESTRICT A, const integer lda,
         scalar* RPY_RESTRICT B, const integer ldb, scalar* RPY_RESTRICT work,
         const integer* RPY_RESTRICT lwork, integer& info) noexcept
    {
        RPY_LAPACK_FUNC(dgels)
        (reinterpret_cast<const char*>(trans), &m, &n, &nrhs, A, &lda, B, &ldb,
         work, lwork, &info);
    }

    static inline void
    gelsy(const integer m, const integer n, const integer nrhs,
          scalar* RPY_RESTRICT A, const integer lda, scalar* RPY_RESTRICT B,
          const integer ldb, integer* RPY_RESTRICT jpvt, const scalar& rcond,
          integer& rank, scalar* RPY_RESTRICT work,
          const integer* RPY_RESTRICT lwork,
          const integer* RPY_UNUSED_VAR rwork, integer& info) noexcept
    {
        RPY_LAPACK_FUNC(dgelsy)
        (&m, &n, &nrhs, A, &lda, B, &ldb, jpvt, &rcond, &rank, work, lwork,
         &info);
    }

    static inline void
    gelss(const integer m, const integer n, const integer nrhs,
          scalar* RPY_RESTRICT A, const integer lda, scalar* RPY_RESTRICT B,
          const integer ldb, scalar* RPY_RESTRICT s, const scalar& rcond,
          integer& rank, scalar* RPY_RESTRICT work,
          const integer* RPY_RESTRICT lwork,
          const integer* RPY_RESTRICT RPY_UNUSED_VAR rwork,
          integer& info) noexcept
    {
        RPY_LAPACK_FUNC(dgelss)
        (&m, &n, &nrhs, A, &lda, B, &ldb, s, &rcond, &rank, work, lwork, &info);
    }

    static inline void
    gelsd(const integer m, const integer n, const integer nrhs,
          scalar* RPY_RESTRICT A, const integer lda, scalar* RPY_RESTRICT B,
          const integer ldb, scalar* RPY_RESTRICT s, const scalar& rcond,
          integer& rank, scalar* RPY_RESTRICT work,
          const integer* RPY_RESTRICT lwork,
          const integer* RPY_RESTRICT RPY_UNUSED_VAR rwork,
          integer* RPY_RESTRICT iwork, integer& info) noexcept
    {
        RPY_LAPACK_FUNC(dgelsd)
        (&m, &n, &nrhs, A, &lda, B, &ldb, s, &rcond, &rank, work, lwork, iwork,
         &info);
    }
};

template <>
struct lapack_funcs<scalars::float_complex, float> {
    using scalar = scalars::float_complex;
    using real_scalar = float;

    static inline void
    gesv(const integer n, const integer nrhs, scalar* RPY_RESTRICT A,
         const integer lda, integer* RPY_RESTRICT ipiv, scalar* RPY_RESTRICT B,
         const integer ldb, integer& info) noexcept
    {
        RPY_LAPACK_FUNC(cgesv)
        (&n, &nrhs, reinterpret_cast<complex32*>(A), &lda, ipiv,
         reinterpret_cast<complex32*>(B), &ldb, &info);
    }

    static inline void
    syev(const char* jobz, blas::BlasUpLo uplo, const integer n,
         scalar* RPY_RESTRICT A, const integer lda, real_scalar* RPY_RESTRICT w,
         scalar* RPY_RESTRICT work, const integer* RPY_RESTRICT lwork,
         real_scalar* RPY_RESTRICT rwork, integer& info) noexcept
    {
        RPY_LAPACK_FUNC(cheev)
        (jobz, reinterpret_cast<const char*>(&uplo), &n,
         reinterpret_cast<complex32*>(A), &lda, w,
         reinterpret_cast<complex32*>(work), lwork, rwork, &info);
    }

    static inline void
    geev(const char* joblv, const char* jobvr, const integer n,
         scalar* RPY_RESTRICT A, const integer lda, scalar* RPY_RESTRICT wr,
         scalar* RPY_RESTRICT RPY_UNUSED_VAR wi, scalar* RPY_RESTRICT vl,
         const integer ldvl, scalar* RPY_RESTRICT vr, const integer ldvr,
         scalar* RPY_RESTRICT work, const integer* RPY_RESTRICT lwork,
         real_scalar* RPY_RESTRICT rwork, integer& info) noexcept
    {
        RPY_LAPACK_FUNC(cgeev)
        (joblv, jobvr, &n, reinterpret_cast<complex32*>(A), &lda,
         reinterpret_cast<complex32*>(wr), reinterpret_cast<complex32*>(vl),
         &ldvl, reinterpret_cast<complex32*>(vr), &ldvr,
         reinterpret_cast<complex32*>(work), lwork, rwork, &info);
    }

    static inline void
    gesvd(const char* jobu, const char* jobvt, const integer m, const integer n,
          scalar* RPY_RESTRICT A, const integer lda,
          real_scalar* RPY_RESTRICT s, scalar* RPY_RESTRICT u,
          const integer ldu, scalar* RPY_RESTRICT vt, const integer ldvt,
          scalar* RPY_RESTRICT work, const integer* RPY_RESTRICT lwork,
          real_scalar* RPY_UNUSED_VAR rwork, integer& info) noexcept
    {
        RPY_LAPACK_FUNC(cgesvd)
        (jobu, jobvt, &m, &n, reinterpret_cast<complex32*>(A), &lda, s,
         reinterpret_cast<complex32*>(u), &ldu,
         reinterpret_cast<complex32*>(vt), &ldvt,
         reinterpret_cast<complex32*>(work), lwork, rwork, &info);
    }

    static inline void
    gesdd(const char* jobz, const integer m, const integer n,
          scalar* RPY_RESTRICT A, const integer lda,
          real_scalar* RPY_RESTRICT s, scalar* RPY_RESTRICT u,
          const integer ldu, scalar* RPY_RESTRICT vt, const integer ldvt,
          scalar* RPY_RESTRICT work, const integer* RPY_RESTRICT lwork,
          integer* RPY_RESTRICT iwork, real_scalar* RPY_RESTRICT rwork,
          integer& info) noexcept
    {
        RPY_LAPACK_FUNC(cgesdd)
        (jobz, &m, &n, reinterpret_cast<complex32*>(A), &lda, s,
         reinterpret_cast<complex32*>(u), &ldu,
         reinterpret_cast<complex32*>(vt), &ldvt,
         reinterpret_cast<complex32*>(work), lwork, rwork, iwork, &info);
    }

    static inline void
    gels(blas::BlasTranspose trans, const integer m, const integer n,
         const integer nrhs, scalar* RPY_RESTRICT A, const integer lda,
         scalar* RPY_RESTRICT B, const integer ldb, scalar* RPY_RESTRICT work,
         const integer* RPY_RESTRICT lwork, integer& info) noexcept
    {
        RPY_LAPACK_FUNC(cgels)
        (reinterpret_cast<const char*>(trans), &m, &n, &nrhs,
         reinterpret_cast<complex32*>(A), &lda, reinterpret_cast<complex32*>(B),
         &ldb, reinterpret_cast<complex32*>(work), lwork, &info);
    }

    static inline void
    gelsy(const integer m, const integer n, const integer nrhs,
          scalar* RPY_RESTRICT A, const integer lda, scalar* RPY_RESTRICT B,
          const integer ldb, integer* RPY_RESTRICT jpvt,
          const real_scalar& rcond, integer& rank, scalar* RPY_RESTRICT work,
          const integer* RPY_RESTRICT lwork, real_scalar* RPY_RESTRICT rwork,
          integer& info) noexcept
    {
        RPY_LAPACK_FUNC(cgelsy)
        (&m, &n, &nrhs, reinterpret_cast<complex32*>(A), &lda,
         reinterpret_cast<complex32*>(B), &ldb, jpvt, &rcond, &rank,
         reinterpret_cast<complex32*>(work), lwork, rwork, &info);
    }

    static inline void
    gelss(const integer m, const integer n, const integer nrhs,
          scalar* RPY_RESTRICT A, const integer lda, scalar* RPY_RESTRICT B,
          const integer ldb, real_scalar* RPY_RESTRICT s,
          const real_scalar& rcond, integer& rank, scalar* RPY_RESTRICT work,
          const integer* RPY_RESTRICT lwork, real_scalar* RPY_RESTRICT rwork,
          integer& info) noexcept
    {
        RPY_LAPACK_FUNC(cgelss)
        (&m, &n, &nrhs, reinterpret_cast<complex32*>(A), &lda,
         reinterpret_cast<complex32*>(B), &ldb, s, &rcond, &rank,
         reinterpret_cast<complex32*>(work), lwork, rwork, &info);
    }

    static inline void
    gelsd(const integer m, const integer n, const integer nrhs,
          scalar* RPY_RESTRICT A, const integer lda, scalar* RPY_RESTRICT B,
          const integer ldb, real_scalar* RPY_RESTRICT s,
          const real_scalar& rcond, integer& rank, scalar* RPY_RESTRICT work,
          const integer* RPY_RESTRICT lwork, real_scalar* RPY_RESTRICT rwork,
          integer* RPY_RESTRICT iwork, integer& info) noexcept
    {
        RPY_LAPACK_FUNC(cgelsd)
        (&m, &n, &nrhs, reinterpret_cast<complex32*>(A), &lda,
         reinterpret_cast<complex32*>(B), &ldb, s, &rcond, &rank,
         reinterpret_cast<complex32*>(work), lwork, rwork, iwork, &info);
    }
};

template <>
struct lapack_funcs<scalars::double_complex, double> {
    using scalar = scalars::double_complex;
    using real_scalar = double;

    static inline void
    gesv(const integer n, const integer nrhs, scalar* RPY_RESTRICT A,
         const integer lda, integer* RPY_RESTRICT ipiv, scalar* RPY_RESTRICT B,
         const integer ldb, integer& info) noexcept
    {
        RPY_LAPACK_FUNC(zgesv)
        (&n, &nrhs, reinterpret_cast<complex64*>(A), &lda, ipiv,
         reinterpret_cast<complex64*>(B), &ldb, &info);
    }

    static inline void
    syev(const char* jobz, blas::BlasUpLo uplo, const integer n,
         scalar* RPY_RESTRICT A, const integer lda, real_scalar* RPY_RESTRICT w,
         scalar* RPY_RESTRICT work, const integer* RPY_RESTRICT lwork,
         real_scalar* RPY_RESTRICT rwork, integer& info) noexcept
    {
        RPY_LAPACK_FUNC(zheev)
        (jobz, reinterpret_cast<const char*>(&uplo), &n,
         reinterpret_cast<complex64*>(A), &lda, w,
         reinterpret_cast<complex64*>(work), lwork, rwork, &info);
    }

    static inline void
    geev(const char* joblv, const char* jobvr, const integer n,
         scalar* RPY_RESTRICT A, const integer lda, scalar* RPY_RESTRICT wr,
         scalar* RPY_RESTRICT RPY_UNUSED_VAR wi, scalar* RPY_RESTRICT vl,
         const integer ldvl, scalar* RPY_RESTRICT vr, const integer ldvr,
         scalar* RPY_RESTRICT work, const integer* RPY_RESTRICT lwork,
         real_scalar* RPY_RESTRICT rwork, integer& info) noexcept
    {
        RPY_LAPACK_FUNC(zgeev)
        (joblv, jobvr, &n, reinterpret_cast<complex64*>(A), &lda,
         reinterpret_cast<complex64*>(wr), reinterpret_cast<complex64*>(vl),
         &ldvl, reinterpret_cast<complex64*>(vr), &ldvr,
         reinterpret_cast<complex64*>(work), lwork, rwork, &info);
    }

    static inline void
    gesvd(const char* jobu, const char* jobvt, const integer m, const integer n,
          scalar* RPY_RESTRICT A, const integer lda,
          real_scalar* RPY_RESTRICT s, scalar* RPY_RESTRICT u,
          const integer ldu, scalar* RPY_RESTRICT vt, const integer ldvt,
          scalar* RPY_RESTRICT work, const integer* RPY_RESTRICT lwork,
          real_scalar* RPY_UNUSED_VAR rwork, integer& info) noexcept
    {
        RPY_LAPACK_FUNC(zgesvd)
        (jobu, jobvt, &m, &n, reinterpret_cast<complex64*>(A), &lda, s,
         reinterpret_cast<complex64*>(u), &ldu,
         reinterpret_cast<complex64*>(vt), &ldvt,
         reinterpret_cast<complex64*>(work), lwork, rwork, &info);
    }

    static inline void
    gesdd(const char* jobz, const integer m, const integer n,
          scalar* RPY_RESTRICT A, const integer lda,
          real_scalar* RPY_RESTRICT s, scalar* RPY_RESTRICT u,
          const integer ldu, scalar* RPY_RESTRICT vt, const integer ldvt,
          scalar* RPY_RESTRICT work, const integer* RPY_RESTRICT lwork,
          integer* RPY_RESTRICT iwork, real_scalar* RPY_RESTRICT rwork,
          integer& info) noexcept
    {
        RPY_LAPACK_FUNC(zgesdd)
        (jobz, &m, &n, reinterpret_cast<complex64*>(A), &lda, s,
         reinterpret_cast<complex64*>(u), &ldu,
         reinterpret_cast<complex64*>(vt), &ldvt,
         reinterpret_cast<complex64*>(work), lwork, rwork, iwork, &info);
    }

    static inline void
    gels(blas::BlasTranspose trans, const integer m, const integer n,
         const integer nrhs, scalar* RPY_RESTRICT A, const integer lda,
         scalar* RPY_RESTRICT B, const integer ldb, scalar* RPY_RESTRICT work,
         const integer* RPY_RESTRICT lwork, integer& info) noexcept
    {
        RPY_LAPACK_FUNC(zgels)
        (reinterpret_cast<const char*>(trans), &m, &n, &nrhs,
         reinterpret_cast<complex64*>(A), &lda, reinterpret_cast<complex64*>(B),
         &ldb, reinterpret_cast<complex64*>(work), lwork, &info);
    }

    static inline void
    gelsy(const integer m, const integer n, const integer nrhs,
          scalar* RPY_RESTRICT A, const integer lda, scalar* RPY_RESTRICT B,
          const integer ldb, integer* RPY_RESTRICT jpvt,
          const real_scalar& rcond, integer& rank, scalar* RPY_RESTRICT work,
          const integer* RPY_RESTRICT lwork, real_scalar* RPY_RESTRICT rwork,
          integer& info) noexcept
    {
        RPY_LAPACK_FUNC(zgelsy)
        (&m, &n, &nrhs, reinterpret_cast<complex64*>(A), &lda,
         reinterpret_cast<complex64*>(B), &ldb, jpvt, &rcond, &rank,
         reinterpret_cast<complex64*>(work), lwork, rwork, &info);
    }

    static inline void
    gelss(const integer m, const integer n, const integer nrhs,
          scalar* RPY_RESTRICT A, const integer lda, scalar* RPY_RESTRICT B,
          const integer ldb, real_scalar* RPY_RESTRICT s,
          const real_scalar& rcond, integer& rank, scalar* RPY_RESTRICT work,
          const integer* RPY_RESTRICT lwork, real_scalar* RPY_RESTRICT rwork,
          integer& info) noexcept
    {
        RPY_LAPACK_FUNC(zgelss)
        (&m, &n, &nrhs, reinterpret_cast<complex64*>(A), &lda,
         reinterpret_cast<complex64*>(B), &ldb, s, &rcond, &rank,
         reinterpret_cast<complex64*>(work), lwork, rwork, &info);
    }

    static inline void
    gelsd(const integer m, const integer n, const integer nrhs,
          scalar* RPY_RESTRICT A, const integer lda, scalar* RPY_RESTRICT B,
          const integer ldb, real_scalar* RPY_RESTRICT s,
          const real_scalar& rcond, integer& rank, scalar* RPY_RESTRICT work,
          const integer* RPY_RESTRICT lwork, real_scalar* RPY_RESTRICT rwork,
          integer* RPY_RESTRICT iwork, integer& info) noexcept
    {
        RPY_LAPACK_FUNC(zgelsd)
        (&m, &n, &nrhs, reinterpret_cast<complex64*>(A), &lda,
         reinterpret_cast<complex64*>(B), &ldb, s, &rcond, &rank,
         reinterpret_cast<complex64*>(work), lwork, rwork, iwork, &info);
    }
};

}// namespace lapack

}// namespace rpy
