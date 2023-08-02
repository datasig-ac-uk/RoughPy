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

#include <roughpy/scalars/owned_scalar_array.h>
#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_array.h>
#include <roughpy/scalars/scalar_blas.h>

#include <sstream>

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
    handle_illegal_parameter_error(const char* method, integer arg)
    {
        std::stringstream ss;
        ss << "invalid argument " << arg << " in call to " << method;
        RPY_THROW(std::invalid_argument, ss.str());
    }

    inline void
    gesv(const integer n, const integer nrhs, scalar* A, const integer lda,
         integer* ipiv, scalar* B, const integer ldb);

    inline void
    syev(const char* jobz, blas::BlasUpLo uplo, const integer n,
         scalar* RPY_RESTRICT A, const integer lda,
         real_scalar* RPY_RESTRICT w);

    inline void
    geev(const char* joblv, const char* jobvr, const integer n,
         scalar* RPY_RESTRICT A, const integer lda, scalar* RPY_RESTRICT wr,
         scalar* RPY_RESTRICT RPY_UNUSED_VAR wi, scalar* RPY_RESTRICT vl,
         const integer ldvl, scalar* RPY_RESTRICT vr, const integer ldvr);

    inline void
    gesvd(const char* jobu, const char* jobvt, const integer m, const integer n,
          scalar* RPY_RESTRICT A, const integer lda,
          real_scalar* RPY_RESTRICT s, scalar* RPY_RESTRICT u,
          const integer ldu, scalar* RPY_RESTRICT vt, const integer ldvt);

    inline void
    gesdd(const char* jobz, const integer m, const integer n,
          scalar* RPY_RESTRICT A, const integer lda,
          real_scalar* RPY_RESTRICT s, scalar* RPY_RESTRICT u,
          const integer ldu, scalar* RPY_RESTRICT vt, const integer ldvt);

    inline void
    gels(blas::BlasTranspose trans, const integer m, const integer n,
         const integer nrhs, scalar* RPY_RESTRICT A, const integer lda,
         scalar* RPY_RESTRICT B, const integer ldb);

    inline void
    gelsy(const integer m, const integer n, const integer nrhs,
          scalar* RPY_RESTRICT A, const integer lda, scalar* RPY_RESTRICT B,
          const integer ldb, integer* RPY_RESTRICT jpvt,
          const real_scalar& rcond, integer& rank);

    inline void
    gelss(const integer m, const integer n, const integer nrhs,
          scalar* RPY_RESTRICT A, const integer lda, scalar* RPY_RESTRICT B,
          const integer ldb, real_scalar* RPY_RESTRICT s,
          const real_scalar& rcond, integer& rank);

    inline void
    gelsd(const integer m, const integer n, const integer nrhs,
          scalar* RPY_RESTRICT A, const integer lda, scalar* RPY_RESTRICT B,
          const integer ldb, real_scalar* RPY_RESTRICT s,
          const real_scalar& rcond, integer& rank);
};

}// namespace lapack

namespace scalars {

template <typename S, typename R>
class StandardLinearAlgebra : public BlasInterface,
                              blas::blas_funcs<S, R>,
                              lapack::lapack_funcs<S, R>
{

    using blas = ::rpy::blas::blas_funcs<S, R>;
    using lapack = ::rpy::lapack::lapack_funcs<S, R>;

    using integer = ::rpy::blas::integer;
    using logical = ::rpy::blas::logical;

    using typename blas::abs_scalar;
    using typename blas::scalar;
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
        const auto alpha = scalar_cast<scalar>(a);

        blas::axpy(
                N, alpha, x.raw_cast<const scalar*>(), 1,
                result.raw_cast<scalar*>(), 1
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
                N, lhs.raw_cast<const scalar*>(), 1,
                rhs.raw_cast<const scalar*>(), 1
        );
        return {type, result};
    }
    Scalar L1Norm(const ScalarArray& vector) override
    {
        auto guard = lock();
        auto N = static_cast<integer>(vector.size());
        auto result = blas::asum(N, vector.raw_cast<const scalar*>(), 1);
        return {type(), result};
    }
    Scalar L2Norm(const ScalarArray& vector) override
    {
        RPY_CHECK(vector.type() == type());
        float result = 0.0;
        auto N = static_cast<integer>(vector.size());
        result = blas::nrm2(N, vector.raw_cast<const scalar*>(), 1);
        return {type(), result};
    }
    Scalar LInfNorm(const ScalarArray& vector) override
    {
        RPY_CHECK(vector.type() == type());
        auto N = static_cast<integer>(vector.size());
        const auto* ptr = vector.raw_cast<const scalar*>();
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
        const auto alp = scalar_cast<scalar>(alpha);
        const auto bet = scalar_cast<scalar>(beta);

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
         * For now, we shall assume that we don't want to transpose the matrix
         * A. In the future we might want to consider transposing based on
         * whether the matrix is in row major or column major format.
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

        blas::gemv(
                layout, transa, m, n, alp, A.raw_cast<const scalar*>(), lda,
                x.raw_cast<const scalar*>(), incx, bet, y.raw_cast<scalar*>(),
                incy
        );
    }
    void
    gemm(ScalarMatrix& C, const ScalarMatrix& A, const ScalarMatrix& B,
         const Scalar& alpha, const Scalar& beta) override
    {
        auto guard = lock();
        type_check(A);
        type_check(B);
        const auto alp = scalar_cast<scalar>(alpha);
        const auto bet = scalar_cast<scalar>(beta);

        integer m = A.nrows();
        integer n = B.ncols();
        integer k = A.ncols();

        const auto lda = A.leading_dimension();
        const auto ldb = B.leading_dimension();
        integer ldc;// initialized below

        BlasLayout layout;
        BlasTranspose transa;
        BlasTranspose transb;

        /*
         * If C is initialized, we use C.layout() to determine whether A and B
         * need to be transposed or not. If C is not initialized, we use
         * A.layout() to set the layout of C and determine whether we should
         * transpose B.
         */
        if (C.type() == nullptr || C.is_null()) {
            // Allocate a new result matrix with dimensions m by n
            C = ScalarMatrix(type(), m, n, A.layout());
            ldc = C.leading_dimension();
            layout = blas::to_blas_layout(A.layout());
            transa = blas::Blas_NoTrans;
            transb = (B.layout() == A.layout()) ? blas::Blas_NoTrans
                                                : blas::Blas_Trans;
        } else {
            type_check(C);
            RPY_CHECK(C.nrows() == m && C.ncols() == n);

            ldc = C.leading_dimension();
            //            layout = to_blas_layout(C.layout());

            transa = (A.layout() == C.layout()) ? blas::Blas_NoTrans
                                                : blas::Blas_Trans;

            transb = (B.layout() == C.layout()) ? blas::Blas_NoTrans
                                                : blas::Blas_Trans;
        }

        blas::gemm(
                layout, transa, transb, m, n, k, alp,
                A.raw_cast<const scalar*>(), lda, B.raw_cast<const scalar*>(),
                ldb, bet, C.raw_cast<scalar*>(), ldc
        );
    }
    void gesv(ScalarMatrix& A, ScalarMatrix& B) override
    {
        auto guard = lock();
        // Solving A*X = B, solution written to B
        type_check(A);
        type_check(B);

        integer n = A.nrows();
        RPY_CHECK(A.ncols() == n);
        integer nrhs = B.ncols();
        matrix_product_check(n, B.nrows());

        const auto layout = blas::to_blas_layout(A.layout());
        const auto lda = A.leading_dimension();
        const auto ldb = B.leading_dimension();
        const auto ldx = ldb;// Assuming X and B have the same shape

        lapack::gesv(
                layout, n, nrhs, A.raw_cast<scalar*>(), B.raw_cast<scalar*>(),
                lda, ldb
        );
    }
    EigenDecomposition syev(ScalarMatrix& A, bool eigenvectors) override
    {
        auto guard = lock();
        const auto layout = blas::to_blas_layout(A.layout());
        const char jobz = eigenvectors ? 'V' : 'N';
        const char range = 'A';
        const char uplo = 'U';// ?

        const auto n = A.nrows();
        const auto lda = A.leading_dimension();
        const scalar* vl = nullptr;
        const scalar* vu = nullptr;
        const integer* il = nullptr;
        const integer* iu = nullptr;

        const float abstol = 0.0;// will be replaced by n*eps*||A||.
        integer m = 0;

        EigenDecomposition result;
        result.Lambda = ScalarMatrix(type(), n, 1, MatrixLayout::ColumnMajor);

        integer ldz = 1;
        scalar* vs = nullptr;
        if (eigenvectors) {
            result.U = ScalarMatrix(type(), n, n, A.layout());
            vs = result.U.raw_cast<scalar*>();
            ldz = result.U.leading_dimension();
        }

        std::vector<integer> isuppz(2 * n);

        lapack::syevr(
                layout, jobz, range, uplo, n, A.raw_cast<scalar*>(), lda, vl,
                vu, il, iu, abstol, &m, result.Lambda.raw_cast<scalar*>(), vs,
                ldz, isuppz.data()
        );
    }
    EigenDecomposition gees(ScalarMatrix& A, bool eigenvectors) override
    {
        auto guard = lock();
        type_check(A);
        RPY_CHECK(!A.is_null());
        RPY_CHECK(A.ncols() == A.nrows());

        const auto layout = blas::to_blas_layout(A.layout());
        const auto n = A.nrows();
        const auto lda = A.leading_dimension();

        const auto jobvs = eigenvectors ? 'V' : 'N';
        const auto sort = 'N';
        LAPACK_S_SELECT2 select = nullptr;

        EigenDecomposition result;

        integer sdim = 0;
        scalar* vs = nullptr;
        integer ldvs = 1;
        if (eigenvectors) {
            result.U = ScalarMatrix(type(), n, n, A.layout());
            ldvs = n;
            vs = result.U.raw_cast<scalar*>();
        }

        /*
         * The eigenvalues of a real, non-symmetric matrix can be complex, so
         * write the real/imaginary parts into two temporary buffers.
         * Afterwards, we can choose whether the result vector has type float or
         * complex<float> depending on whether any of the eigenvalues are
         * complex.
         */
        std::vector<scalar> ev_real(n);
        std::vector<scalar> ev_imag(n);

        lapack::gees(
                layout, jobvs, sort, select, n, A.raw_cast<scalar*>(), lda,
                &sdim, ev_real.data(), ev_imag.data(), vs, ldvs
        );

        bool any_complex = std::any_of(
                ev_imag.begin(), ev_imag.end(),
                [](const scalar& v) { return v != 0.0f; }
        );

        if (any_complex) {
            result.Lambda = ScalarMatrix(type(), n, 1);
            ////// TODO: Implement complex number stuff
        } else {
            result.Lambda = ScalarMatrix(type(), n, 1);
            std::copy_n(ev_real.begin(), n, result.Lambda.raw_cast<scalar*>());
        }

        return result;
    }
    SingularValueDecomposition
    gesvd(ScalarMatrix& A, bool return_U, bool return_VT) override
    {
        auto guard = lock();
        type_check(A);
        RPY_CHECK(!A.is_null());

        const auto layout = blas::to_blas_layout(A.layout());
        const auto m = A.nrows();
        const auto n = A.ncols();
        const auto lda = A.leading_dimension();

        integer ldu = 1;
        integer ldvt = 1;
        char jobu = 'N';
        char jobvt = 'N';

        SingularValueDecomposition result;
        result.Sigma = ScalarMatrix(type(), n, 1, MatrixLayout::ColumnMajor);

        scalar* u = nullptr;
        scalar* vt = nullptr;

        if (return_U) {
            jobu = 'A';
            result.U = ScalarMatrix(type(), m, m);
            ldu = m;
            u = result.U.raw_cast<scalar*>();
        }

        if (return_VT) {
            jobvt = 'A';
            result.VHermitian = ScalarMatrix(type(), n, n);
            ldvt = n;
            vt = result.VHermitian.raw_cast<scalar*>();
        }

        //        m_workspace.clear();
        //        m_workspace.resize(std::min(m, n) - 2);

        auto info = lapack::gesvd(
                layout, jobu, jobvt, m, n, A.raw_cast<scalar*>(), lda,
                result.Sigma.raw_cast<scalar*>(), u, ldu, vt, ldvt
                //                m_workspace.data()
        );

        // Handle errors that might have happened in LAPACK
        check_and_report_errors(info);

        return result;
    }
    SingularValueDecomposition
    gesdd(ScalarMatrix& A, bool return_U, bool return_VT) override
    {
        auto guard = lock();
        type_check(A);
        RPY_CHECK(!A.is_null());

        const auto layout = blas::to_blas_layout(A.layout());
        const auto m = A.nrows();
        const auto n = A.ncols();
        const auto lda = A.leading_dimension();

        integer ldu = 1;
        integer ldvt = 1;
        char jobz = 'N';

        SingularValueDecomposition result;
        result.Sigma = ScalarMatrix(type(), n, 1, MatrixLayout::ColumnMajor);

        scalar* u = nullptr;
        scalar* vt = nullptr;

        /*
         * Unlike sgesvd, we can either get both U and VT or neither. We
         * maintain the same interface though, but just populate both if either
         * are requested.
         */
        if (return_U || return_VT) {
            jobz = 'A';
            result.U = ScalarMatrix(type(), m, m);
            ldu = m;
            u = result.U.raw_cast<scalar*>();
            result.VHermitian = ScalarMatrix(type(), n, n);
            ldvt = n;
            vt = result.VHermitian.raw_cast<scalar*>();
        }

        integer info = 0;
        lapack::gesdd(
                layout, jobz, m, n, A.raw_cast<scalar*>(), lda,
                result.Sigma.raw_cast<scalar*>(), u, ldu, vt, ldvt, info
        );

        // Handle errors that might have happened in LAPACK
        // check_and_report_errors(info);

        return result;
    }
    void gels(ScalarMatrix& A, ScalarMatrix& b) override {}
    ScalarMatrix gelsy(ScalarMatrix& A, ScalarMatrix& b) override {}
    ScalarMatrix gelss(ScalarMatrix& A, ScalarMatrix& b) override {}
    ScalarMatrix gelsd(ScalarMatrix& A, ScalarMatrix& b) override {}
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
    std::vector<float> m_work;
    std::vector<integer> m_iwork;
    integer lwork;

    void reset_workspace() {
        lwork = -1;
        m_work.resize(1);
        m_iwork.resize(1);
    }

    void resize_workspace(bool iwork=false)
    {
        lwork = static_cast<integer>(m_work[0]);
        m_work.resize(lwork);
        if (iwork) {
            m_iwork.resize(m_iwork[0]);
        }
    }
};

template <>
struct lapack_func_workspace<double, double> {
    std::vector<double> m_work;
    std::vector<integer> m_iwork;
    integer lwork;

    void reset_workspace() {
        lwork = -1;
        m_work.resize(1);
        m_iwork.resize(1);
    }
    void resize_workspace(bool iwork=false)
    {
        lwork = static_cast<integer>(m_work[0]);
        m_work.resize(lwork);
        if (iwork) {
            m_iwork.resize(m_iwork[0]);
        }
    }
};

template <>
struct lapack_func_workspace<scalars::float_complex, float> {
    std::vector<complex32> m_work;
    std::vector<float> m_rwork;
    std::vector<integer> m_iwork;
    integer lwork;

    void reset_workspace() {
        lwork = -1;
        m_work.resize(1);
        m_iwork.resize(1);
        m_rwork.resize(1);
    }

    void resize_workspace(bool rwork=false, bool iwork=false)
    {
        lwork = static_cast<integer>(m_work[0].real);
        m_work.resize(lwork);
        if (iwork) {
            m_iwork.resize(m_iwork[0]);
        }
        if (rwork) {
            m_rwork.resize(static_cast<integer>(m_rwork[0]));
        }
    }
};

template <>
struct lapack_func_workspace<scalars::double_complex, double> {
    std::vector<complex64> m_work;
    std::vector<double> m_rwork;
    std::vector<integer> m_iwork;
    integer lwork;

    void reset_workspace() {
        lwork = -1;
        m_work.resize(1);
        m_iwork.resize(1);
        m_rwork.resize(1);
    }

    void resize_workspace(bool rwork=false, bool iwork=false)
    {
        lwork = static_cast<integer>(m_work[0].real);
        m_work.resize(lwork);
        if (iwork) {
            m_iwork.resize(m_iwork[0]);
        }
        if (rwork) {
            m_rwork.resize(static_cast<integer>(m_rwork[0]));
        }
    }
};

template <>
void lapack_funcs<float, float>::gesv(
        const integer n, const integer nrhs, float* A, const integer lda,
        integer* ipiv, float* B, const integer ldb
)
{
    integer info = 0;
    RPY_LAPACK_FUNC(sgesv)(&n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
    if (info < 0) {
        handle_illegal_parameter_error("gesv", -info);
    } else if (info > 0) {
        std::stringstream ss;
        ss << "component" << info
           << " on the diagonal of U is zero so the matrix is singular";
        RPY_THROW(std::runtime_error, ss.str());
    }
}
template <>
void lapack_funcs<float, float>::syev(
        const char* jobz, blas::BlasUpLo uplo, const integer n, float* A,
        const integer lda, float* w
)
{
    const auto* uplo_ = reinterpret_cast<const char*>(&uplo);
    integer info = 0;

    reset_workspace();
    RPY_LAPACK_FUNC(ssyev)
    (jobz, uplo_, &n, A, &lda, w, m_work.data(), &lwork, &info);
    RPY_CHECK(info == 0);
    resize_workspace();

    RPY_LAPACK_FUNC(ssyev)
    (jobz, uplo_, &n, A, &lda, w, m_work.data(), &lwork, &info);
    if (info < 0) {
        handle_illegal_parameter_error("syev", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "the eigenvalues failed to converge");
    }
}
template <>
void lapack_funcs<float, float>::geev(
        const char* jobvl, const char* jobvr, const integer n, float* A,
        const integer lda, float* wr, float* wi, float* vl, const integer ldvl,
        float* vr, const integer ldvr
)
{
    integer info = 0;

    reset_workspace();
    RPY_LAPACK_FUNC(sgeev)
    (jobvl, jobvr, &n, A, &lda, wr, wi, vl, &ldvl, vr, &ldvr, m_work.data(),
     &lwork, &info);
    RPY_CHECK(info == 0);
    resize_workspace();

    RPY_LAPACK_FUNC(sgeev)
    (jobvl, jobvr, &n, A, &lda, wr, wi, vl, &ldvl, vr, &ldvr, m_work.data(),
     &lwork, &info);

    if (info < 0) {
        handle_illegal_parameter_error("geev", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "the eigenvalues failed to converge");
    }
}
template <>
void lapack_funcs<float, float>::gesvd(
        const char* jobu, const char* jobvt, const integer m, const integer n,
        float* A, const integer lda, float* s, float* u, const integer ldu,
        float* vt, const integer ldvt
)
{
    integer info = 0;

    reset_workspace();
    RPY_LAPACK_FUNC(sgesvd)
    (jobu, jobvt, &m, &n, A, &lda, s, u, &ldu, vt, &ldvt, m_work.data(), &lwork,
     &info);
    RPY_CHECK(info == 0);
    resize_workspace();

    RPY_LAPACK_FUNC(sgesvd)
    (jobu, jobvt, &m, &n, A, &lda, s, u, &ldu, vt, &ldvt, m_work.data(), &lwork,
     &info);
    if (info < 0) {
        handle_illegal_parameter_error("gesvd", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "singular values failed to converge");
    }
}
template <>
void lapack_funcs<float, float>::gesdd(
        const char* jobz, const integer m, const integer n, float* A,
        const integer lda, float* s, float* u, const integer ldu, float* vt,
        const integer ldvt
)
{
    integer info = 0;
    auto* a = A;
    auto* u_ = u;
    auto* vt_ = vt;

    reset_workspace();
    RPY_LAPACK_FUNC(sgesdd)
    (jobz, &m, &n, a, &lda, s, u_, &ldu, vt_, &ldvt, m_work.data(), &lwork,
     m_iwork.data(), &info);
    RPY_CHECK(info == 0);
    resize_workspace(true);

    RPY_LAPACK_FUNC(sgesdd)
    (jobz, &m, &n, a, &lda, s, u_, &ldu, vt_, &ldvt, m_work.data(), &lwork,
     m_iwork.data(), &info);
    if (info == -4) {
        RPY_THROW(std::invalid_argument, "matrix A contains a NaN value");
    } else if (info < 0) {
        handle_illegal_parameter_error("gesdd", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "singular values failed to converge");
    }
}
template <>
void lapack_funcs<float, float>::gels(
        blas::BlasTranspose trans, const integer m, const integer n,
        const integer nrhs, float* A, const integer lda, float* B,
        const integer ldb
)
{
    integer info = 0;
    const auto* trans_ = reinterpret_cast<const char*>(&trans);
    auto* a = A;
    auto* b = B;

    reset_workspace();
    RPY_LAPACK_FUNC(sgels)
    (trans_, &m, &n, &nrhs, a, &lda, b, &ldb, m_work.data(), &lwork, &info);
    RPY_CHECK(info == 0);
    resize_workspace();

    RPY_LAPACK_FUNC(sgels)
    (trans_, &m, &n, &nrhs, a, &lda, b, &ldb, m_work.data(), &lwork, &info);
    if (info < 0) {
        handle_illegal_parameter_error("gels", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "matrix does not have full rank");
    }
}
template <>
void lapack_funcs<float, float>::gelsy(
        const integer m, const integer n, const integer nrhs, float* A,
        const integer lda, float* B, const integer ldb, integer* jpvt,
        const float& rcond, integer& rank
)
{
    integer info = 0;
    auto* a = A;
    auto* b = B;
    auto* rcond_ = &rcond;

    reset_workspace();
    RPY_LAPACK_FUNC(sgelsy)
    (&m, &n, &nrhs, a, &lda, b, &ldb, jpvt, rcond_, &rank, m_work.data(),
     &lwork, &info);
    RPY_CHECK(info == 0);
    resize_workspace();

    RPY_LAPACK_FUNC(sgelsy)
    (&m, &n, &nrhs, a, &lda, b, &ldb, jpvt, rcond_, &rank, m_work.data(),
     &lwork, &info);
    if (info < 0) {
        handle_illegal_parameter_error("gelsy", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "matrix does not have full rank");
    }
}
template <>
void lapack_funcs<float, float>::gelss(
        const integer m, const integer n, const integer nrhs, float* A,
        const integer lda, float* B, const integer ldb, float* s,
        const float& rcond, integer& rank
)
{
    integer info = 0;
    auto* a = A;
    auto* b = B;
    const auto* rcond_ = &rcond;

    reset_workspace();
    RPY_LAPACK_FUNC(sgelss)
    (&m, &n, &nrhs, a, &lda, b, &ldb, s, rcond_, &rank, m_work.data(), &lwork,
     &info);
    RPY_CHECK(info == 0);
    resize_workspace();

    RPY_LAPACK_FUNC(sgelss)
    (&m, &n, &nrhs, a, &lda, b, &ldb, s, rcond_, &rank, m_work.data(), &lwork,
     &info);
    if (info < 0) {
        handle_illegal_parameter_error("gelss", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "algorithm for computing svd failed");
    }
}
template <>
void lapack_funcs<float, float>::gelsd(
        const integer m, const integer n, const integer nrhs, float* A,
        const integer lda, float* B, const integer ldb, float* s,
        const float& rcond, integer& rank
)
{
    integer info = 0;

    auto* a = A;
    auto* b = B;
    const auto* rcond_ = &rcond;

    reset_workspace();
    RPY_LAPACK_FUNC(sgelsd)
    (&m, &n, &nrhs, a, &lda, b, &ldb, s, rcond_, &rank, m_work.data(), &lwork,
     m_iwork.data(), &info);
    RPY_CHECK(info == 0);
    resize_workspace(true);

    RPY_LAPACK_FUNC(sgelsd)
    (&m, &n, &nrhs, a, &lda, b, &ldb, s, rcond_, &rank, m_work.data(), &lwork,
     m_iwork.data(), &info);
    if (info < 0) {
        handle_illegal_parameter_error("gelsd", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "algorithm for computing svd failed");
    }
}

template <>
void lapack_funcs<double, double>::gesv(
        const integer n, const integer nrhs, double* A, const integer lda,
        integer* ipiv, double* B, const integer ldb
)
{
    integer info = 0;
    RPY_LAPACK_FUNC(dgesv)(&n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
    if (info < 0) {
        handle_illegal_parameter_error("gesv", -info);
    } else if (info > 0) {
        std::stringstream ss;
        ss << "component" << info
           << " on the diagonal of U is zero so the matrix is singular";
        RPY_THROW(std::runtime_error, ss.str());
    }
}
template <>
void lapack_funcs<double, double>::syev(
        const char* jobz, blas::BlasUpLo uplo, const integer n, double* A,
        const integer lda, double* w
)
{
    const auto* uplo_ = reinterpret_cast<const char*>(&uplo);
    integer info = 0;

    reset_workspace();
    RPY_LAPACK_FUNC(dsyev)
    (jobz, uplo_, &n, A, &lda, w, m_work.data(), &lwork, &info);
    resize_workspace();

    RPY_LAPACK_FUNC(dsyev)
    (jobz, uplo_, &n, A, &lda, w, m_work.data(), &lwork, &info);
    if (info < 0) {
        handle_illegal_parameter_error("syev", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "the eigenvalues failed to converge");
    }
}
template <>
void lapack_funcs<double, double>::geev(
        const char* jobvl, const char* jobvr, const integer n, double* A,
        const integer lda, double* wr, double* wi, double* vl,
        const integer ldvl, double* vr, const integer ldvr
)
{
    integer info = 0;

    reset_workspace();
    RPY_LAPACK_FUNC(dgeev)
    (jobvl, jobvr, &n, A, &lda, wr, wi, vl, &ldvl, vr, &ldvr, m_work.data(),
     &lwork, &info);
    RPY_CHECK(info == 0);
    resize_workspace();

    RPY_LAPACK_FUNC(dgeev)
    (jobvl, jobvr, &n, A, &lda, wr, wi, vl, &ldvl, vr, &ldvr, m_work.data(),
     &lwork, &info);

    if (info < 0) {
        handle_illegal_parameter_error("geev", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "the eigenvalues failed to converge");
    }
}
template <>
void lapack_funcs<double, double>::gesvd(
        const char* jobu, const char* jobvt, const integer m, const integer n,
        double* A, const integer lda, double* s, double* u, const integer ldu,
        double* vt, const integer ldvt
)
{
    integer info = 0;

    reset_workspace();
    RPY_LAPACK_FUNC(dgesvd)
    (jobu, jobvt, &m, &n, A, &lda, s, u, &ldu, vt, &ldvt, m_work.data(), &lwork,
     &info);
    RPY_CHECK(info == 0);
    resize_workspace();

    RPY_LAPACK_FUNC(dgesvd)
    (jobu, jobvt, &m, &n, A, &lda, s, u, &ldu, vt, &ldvt, m_work.data(), &lwork,
     &info);
    if (info < 0) {
        handle_illegal_parameter_error("gesvd", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "singular values failed to converge");
    }
}
template <>
void lapack_funcs<double, double>::gesdd(
        const char* jobz, const integer m, const integer n, double* A,
        const integer lda, double* s, double* u, const integer ldu, double* vt,
        const integer ldvt
)
{
    integer info = 0;
    auto* a = A;
    auto* u_ = u;
    auto* vt_ = vt;

    reset_workspace();
    RPY_LAPACK_FUNC(dgesdd)
    (jobz, &m, &n, a, &lda, s, u_, &ldu, vt_, &ldvt, m_work.data(), &lwork,
     m_iwork.data(), &info);
    RPY_CHECK(info == 0);
    resize_workspace(true);

    RPY_LAPACK_FUNC(dgesdd)
    (jobz, &m, &n, a, &lda, s, u_, &ldu, vt_, &ldvt, m_work.data(), &lwork,
     m_iwork.data(), &info);
    if (info == -4) {
        RPY_THROW(std::invalid_argument, "matrix A contains a NaN value");
    } else if (info < 0) {
        handle_illegal_parameter_error("gesdd", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "singular values failed to converge");
    }
}
template <>
void lapack_funcs<double, double>::gels(
        blas::BlasTranspose trans, const integer m, const integer n,
        const integer nrhs, double* A, const integer lda, double* B,
        const integer ldb
)
{
    integer info = 0;
    const auto* trans_ = reinterpret_cast<const char*>(&trans);
    auto* a = A;
    auto* b = B;

    reset_workspace();
    RPY_LAPACK_FUNC(dgels)
    (trans_, &m, &n, &nrhs, a, &lda, b, &ldb, m_work.data(), &lwork, &info);
    RPY_CHECK(info == 0);
    resize_workspace();

    RPY_LAPACK_FUNC(dgels)
    (trans_, &m, &n, &nrhs, a, &lda, b, &ldb, m_work.data(), &lwork, &info);
    if (info < 0) {
        handle_illegal_parameter_error("gels", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "matrix does not have full rank");
    }
}
template <>
void lapack_funcs<double, double>::gelsy(
        const integer m, const integer n, const integer nrhs, double* A,
        const integer lda, double* B, const integer ldb, integer* jpvt,
        const double& rcond, integer& rank
)
{
    integer info = 0;
    auto* a = A;
    auto* b = B;
    auto* rcond_ = &rcond;

    reset_workspace();
    RPY_LAPACK_FUNC(dgelsy)
    (&m, &n, &nrhs, a, &lda, b, &ldb, jpvt, rcond_, &rank, m_work.data(),
     &lwork, &info);
    RPY_CHECK(info == 0);
    resize_workspace();

    RPY_LAPACK_FUNC(dgelsy)
    (&m, &n, &nrhs, a, &lda, b, &ldb, jpvt, rcond_, &rank, m_work.data(),
     &lwork, &info);
    if (info < 0) {
        handle_illegal_parameter_error("gelsy", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "matrix does not have full rank");
    }
}
template <>
void lapack_funcs<double, double>::gelss(
        const integer m, const integer n, const integer nrhs, double* A,
        const integer lda, double* B, const integer ldb, double* s,
        const double& rcond, integer& rank
)
{
    integer info = 0;
    auto* a = A;
    auto* b = B;
    const auto* rcond_ = &rcond;

    reset_workspace();
    RPY_LAPACK_FUNC(dgelss)
    (&m, &n, &nrhs, a, &lda, b, &ldb, s, rcond_, &rank, m_work.data(), &lwork,
     &info);
    RPY_CHECK(info == 0);
    resize_workspace();

    RPY_LAPACK_FUNC(dgelss)
    (&m, &n, &nrhs, a, &lda, b, &ldb, s, rcond_, &rank, m_work.data(), &lwork,
     &info);
    if (info < 0) {
        handle_illegal_parameter_error("gelss", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "algorithm for computing svd failed");
    }
}
template <>
void lapack_funcs<double, double>::gelsd(
        const integer m, const integer n, const integer nrhs, double* A,
        const integer lda, double* B, const integer ldb, double* s,
        const double& rcond, integer& rank
)
{
    integer info = 0;

    auto* a = A;
    auto* b = B;
    const auto* rcond_ = &rcond;

    reset_workspace();
    RPY_LAPACK_FUNC(dgelsd)
    (&m, &n, &nrhs, a, &lda, b, &ldb, s, rcond_, &rank, m_work.data(), &lwork,
     m_iwork.data(), &info);
    RPY_CHECK(info == 0);
    resize_workspace(true);

    RPY_LAPACK_FUNC(dgelsd)
    (&m, &n, &nrhs, a, &lda, b, &ldb, s, rcond_, &rank, m_work.data(), &lwork,
     m_iwork.data(), &info);
    if (info < 0) {
        handle_illegal_parameter_error("gelsd", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "algorithm for computing svd failed");
    }
}

template <>
void lapack_funcs<scalars::float_complex, float>::gesv(
        const integer n, const integer nrhs, scalars::float_complex* A,
        const integer lda, integer* ipiv, scalars::float_complex* B,
        const integer ldb
)
{
    integer info = 0;
    RPY_LAPACK_FUNC(cgesv)
    (&n, &nrhs, reinterpret_cast<complex32*>(A), &lda, ipiv,
     reinterpret_cast<complex32*>(B), &ldb, &info);
    if (info < 0) {
        handle_illegal_parameter_error("gesv", -info);
    } else if (info > 0) {
        std::stringstream ss;
        ss << "component" << info
           << " on the diagonal of U is zero so the matrix is singular";
        RPY_THROW(std::runtime_error, ss.str());
    }
}
template <>
void lapack_funcs<scalars::float_complex, float>::syev(
        const char* jobz, blas::BlasUpLo uplo, const integer n,
        scalars::float_complex* A, const integer lda, float* w
)
{
    const auto* uplo_ = reinterpret_cast<const char*>(&uplo);
    auto* a = reinterpret_cast<complex32*>(A);

    integer info = 0;
    reset_workspace();
    RPY_LAPACK_FUNC(cheev)
    (jobz, uplo_, &n, a, &lda, w, m_work.data(), &lwork, m_rwork.data(), &info);
    resize_workspace(true);

    RPY_LAPACK_FUNC(cheev)
    (jobz, uplo_, &n, a, &lda, w, m_work.data(), &lwork, m_rwork.data(), &info);
    if (info < 0) {
        handle_illegal_parameter_error("syev", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "the eigenvalues failed to converge");
    }
}
template <>
void lapack_funcs<scalars::float_complex, float>::geev(
        const char* jobvl, const char* jobvr, const integer n,
        scalars::float_complex* A, const integer lda,
        scalars::float_complex* wr, scalars::float_complex* RPY_UNUSED_VAR wi,
        scalars::float_complex* vl, const integer ldvl,
        scalars::float_complex* vr, const integer ldvr
)
{
    integer info = 0;

    auto* a = reinterpret_cast<complex32*>(A);
    auto* w = reinterpret_cast<complex32*>(wr);
    auto* vl_ = reinterpret_cast<complex32*>(vl);
    auto* vr_ = reinterpret_cast<complex32*>(vr);

    reset_workspace();
    RPY_LAPACK_FUNC(cgeev)
    (jobvl, jobvr, &n, a, &lda, w, vl_, &ldvl, vr_, &ldvr, m_work.data(),
     &lwork, m_rwork.data(), &info);
    RPY_CHECK(info == 0);
    resize_workspace(true);

    RPY_LAPACK_FUNC(cgeev)
    (jobvl, jobvr, &n, a, &lda, w, vl_, &ldvl, vr_, &ldvr, m_work.data(),
     &lwork, m_rwork.data(), &info);

    if (info < 0) {
        handle_illegal_parameter_error("geev", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "the eigenvalues failed to converge");
    }
}
template <>
void lapack_funcs<scalars::float_complex, float>::gesvd(
        const char* jobu, const char* jobvt, const integer m, const integer n,
        scalars::float_complex* A, const integer lda, float* s,
        scalars::float_complex* u, const integer ldu,
        scalars::float_complex* vt, const integer ldvt
)
{
    integer info = 0;

    auto* a = reinterpret_cast<complex32*>(A);
    auto* u_ = reinterpret_cast<complex32*>(u);
    auto* vt_ = reinterpret_cast<complex32*>(vt);

    reset_workspace();
    RPY_LAPACK_FUNC(cgesvd)
    (jobu, jobvt, &m, &n, a, &lda, s, u_, &ldu, vt_, &ldvt, m_work.data(),
     &lwork, m_rwork.data(), &info);
    RPY_CHECK(info == 0);
    resize_workspace(true);

    RPY_LAPACK_FUNC(cgesvd)
    (jobu, jobvt, &m, &n, a, &lda, s, u_, &ldu, vt_, &ldvt, m_work.data(),
     &lwork, m_rwork.data(), &info);
    if (info < 0) {
        handle_illegal_parameter_error("gesvd", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "singular values failed to converge");
    }
}
template <>
void lapack_funcs<scalars::float_complex, float>::gesdd(
        const char* jobz, const integer m, const integer n,
        scalars::float_complex* A, const integer lda, float* s,
        scalars::float_complex* u, const integer ldu,
        scalars::float_complex* vt, const integer ldvt
)
{
    integer info = 0;
    auto* a = reinterpret_cast<complex32*>(A);
    auto* u_ = reinterpret_cast<complex32*>(u);
    auto* vt_ = reinterpret_cast<complex32*>(vt);

    reset_workspace();
    RPY_LAPACK_FUNC(cgesdd)
    (jobz, &m, &n, a, &lda, s, u_, &ldu, vt_, &ldvt, m_work.data(), &lwork,
     m_rwork.data(), m_iwork.data(), &info);
    RPY_CHECK(info == 0);
    resize_workspace(true, true);


    RPY_LAPACK_FUNC(cgesdd)
    (jobz, &m, &n, a, &lda, s, u_, &ldu, vt_, &ldvt, m_work.data(), &lwork,
     m_rwork.data(), m_iwork.data(), &info);
    if (info == -4) {
        RPY_THROW(std::invalid_argument, "matrix A contains a NaN value");
    } else if (info < 0) {
        handle_illegal_parameter_error("gesdd", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "singular values failed to converge");
    }
}
template <>
void lapack_funcs<scalars::float_complex, float>::gels(
        blas::BlasTranspose trans, const integer m, const integer n,
        const integer nrhs, scalars::float_complex* A, const integer lda,
        scalars::float_complex* B, const integer ldb
)
{
    integer info = 0;
    const auto* trans_ = reinterpret_cast<const char*>(&trans);
    auto* a = reinterpret_cast<complex32*>(A);
    auto* b = reinterpret_cast<complex32*>(B);

    reset_workspace();
    RPY_LAPACK_FUNC(cgels)
    (trans_, &m, &n, &nrhs, a, &lda, b, &ldb, m_work.data(), &lwork, &info);
    RPY_CHECK(info == 0);
    resize_workspace();

    RPY_LAPACK_FUNC(cgels)
    (trans_, &m, &n, &nrhs, a, &lda, b, &ldb, m_work.data(), &lwork, &info);
    if (info < 0) {
        handle_illegal_parameter_error("gels", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "matrix does not have full rank");
    }
}
template <>
void lapack_funcs<scalars::float_complex, float>::gelsy(
        const integer m, const integer n, const integer nrhs,
        scalars::float_complex* A, const integer lda, scalars::float_complex* B,
        const integer ldb, integer* jpvt, const float& rcond, integer& rank
)
{
    integer info = 0;
    auto* a = reinterpret_cast<complex32*>(A);
    auto* b = reinterpret_cast<complex32*>(B);
    auto* rcond_ = &rcond;

    reset_workspace();
    RPY_LAPACK_FUNC(cgelsy)
    (&m, &n, &nrhs, a, &lda, b, &ldb, jpvt, rcond_, &rank, m_work.data(),
     &lwork, m_rwork.data(), &info);
    RPY_CHECK(info == 0);
    resize_workspace(true);

    RPY_LAPACK_FUNC(cgelsy)
    (&m, &n, &nrhs, a, &lda, b, &ldb, jpvt, rcond_, &rank, m_work.data(),
     &lwork, m_rwork.data(), &info);
    if (info < 0) {
        handle_illegal_parameter_error("gelsy", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "matrix does not have full rank");
    }
}
template <>
void lapack_funcs<scalars::float_complex, float>::gelss(
        const integer m, const integer n, const integer nrhs,
        scalars::float_complex* A, const integer lda, scalars::float_complex* B,
        const integer ldb, float* s, const float& rcond, integer& rank
)
{
    integer info = 0;
    auto* a = reinterpret_cast<complex32*>(A);
    auto* b = reinterpret_cast<complex32*>(B);
    const auto* rcond_ = &rcond;

    reset_workspace();
    RPY_LAPACK_FUNC(cgelss)
    (&m, &n, &nrhs, a, &lda, b, &ldb, s, rcond_, &rank, m_work.data(), &lwork,
     m_rwork.data(), &info);
    RPY_CHECK(info == 0);
    resize_workspace(true);

    RPY_LAPACK_FUNC(cgelss)
    (&m, &n, &nrhs, a, &lda, b, &ldb, s, rcond_, &rank, m_work.data(), &lwork,
     m_rwork.data(), &info);
    if (info < 0) {
        handle_illegal_parameter_error("gelss", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "algorithm for computing svd failed");
    }
}
template <>
void lapack_funcs<scalars::float_complex, float>::gelsd(
        const integer m, const integer n, const integer nrhs,
        scalars::float_complex* A, const integer lda, scalars::float_complex* B,
        const integer ldb, float* s, const float& rcond, integer& rank
)
{
    integer info = 0;

    auto* a = reinterpret_cast<complex32*>(A);
    auto* b = reinterpret_cast<complex32*>(B);
    const auto* rcond_ = &rcond;

    reset_workspace();
    RPY_LAPACK_FUNC(cgelsd)
    (&m, &n, &nrhs, a, &lda, b, &ldb, s, rcond_, &rank, m_work.data(), &lwork,
     m_rwork.data(), m_iwork.data(), &info);
    RPY_CHECK(info == 0);
    resize_workspace(true, true);

    RPY_LAPACK_FUNC(cgelsd)
    (&m, &n, &nrhs, a, &lda, b, &ldb, s, rcond_, &rank, m_work.data(), &lwork,
     m_rwork.data(), m_iwork.data(), &info);
    if (info < 0) {
        handle_illegal_parameter_error("gelsd", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "algorithm for computing svd failed");
    }
}

template <>
void lapack_funcs<scalars::double_complex, double>::gesv(
        const integer n, const integer nrhs, scalars::double_complex* A,
        const integer lda, integer* ipiv, scalars::double_complex* B,
        const integer ldb
)
{
    integer info = 0;
    RPY_LAPACK_FUNC(zgesv)
    (&n, &nrhs, reinterpret_cast<complex64*>(A), &lda, ipiv,
     reinterpret_cast<complex64*>(B), &ldb, &info);
    if (info < 0) {
        handle_illegal_parameter_error("gesv", -info);
    } else if (info > 0) {
        std::stringstream ss;
        ss << "component" << info
           << " on the diagonal of U is zero so the matrix is singular";
        RPY_THROW(std::runtime_error, ss.str());
    }
}
template <>
void lapack_funcs<scalars::double_complex, double>::syev(
        const char* jobz, blas::BlasUpLo uplo, const integer n,
        scalars::double_complex* A, const integer lda, double* w
)
{
    const auto* uplo_ = reinterpret_cast<const char*>(&uplo);
    auto* a = reinterpret_cast<complex64*>(A);

    integer info = 0;
    reset_workspace();
    RPY_LAPACK_FUNC(zheev)
    (jobz, uplo_, &n, a, &lda, w, m_work.data(), &lwork, m_rwork.data(), &info);
    resize_workspace(true);

    RPY_LAPACK_FUNC(zheev)
    (jobz, uplo_, &n, a, &lda, w, m_work.data(), &lwork, m_rwork.data(), &info);
    if (info < 0) {
        handle_illegal_parameter_error("syev", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "the eigenvalues failed to converge");
    }
}
template <>
void lapack_funcs<scalars::double_complex, double>::geev(
        const char* jobvl, const char* jobvr, const integer n,
        scalars::double_complex* A, const integer lda,
        scalars::double_complex* wr, scalars::double_complex* RPY_UNUSED_VAR wi,
        scalars::double_complex* vl, const integer ldvl,
        scalars::double_complex* vr, const integer ldvr
)
{
    integer info = 0;

    auto* a = reinterpret_cast<complex64*>(A);
    auto* w = reinterpret_cast<complex64*>(wr);
    auto* vl_ = reinterpret_cast<complex64*>(vl);
    auto* vr_ = reinterpret_cast<complex64*>(vr);

    reset_workspace();
    RPY_LAPACK_FUNC(zgeev)
    (jobvl, jobvr, &n, a, &lda, w, vl_, &ldvl, vr_, &ldvr, m_work.data(),
     &lwork, m_rwork.data(), &info);
    RPY_CHECK(info == 0);
    resize_workspace(true);

    RPY_LAPACK_FUNC(zgeev)
    (jobvl, jobvr, &n, a, &lda, w, vl_, &ldvl, vr_, &ldvr, m_work.data(),
     &lwork, m_rwork.data(), &info);

    if (info < 0) {
        handle_illegal_parameter_error("geev", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "the eigenvalues failed to converge");
    }
}
template <>
void lapack_funcs<scalars::double_complex, double>::gesvd(
        const char* jobu, const char* jobvt, const integer m, const integer n,
        scalars::double_complex* A, const integer lda, double* s,
        scalars::double_complex* u, const integer ldu,
        scalars::double_complex* vt, const integer ldvt
)
{
    integer info = 0;

    auto* a = reinterpret_cast<complex64*>(A);
    auto* u_ = reinterpret_cast<complex64*>(u);
    auto* vt_ = reinterpret_cast<complex64*>(vt);

    reset_workspace();
    RPY_LAPACK_FUNC(zgesvd)
    (jobu, jobvt, &m, &n, a, &lda, s, u_, &ldu, vt_, &ldvt, m_work.data(),
     &lwork, m_rwork.data(), &info);
    RPY_CHECK(info == 0);
    resize_workspace(true);

    RPY_LAPACK_FUNC(zgesvd)
    (jobu, jobvt, &m, &n, a, &lda, s, u_, &ldu, vt_, &ldvt, m_work.data(),
     &lwork, m_rwork.data(), &info);
    if (info < 0) {
        handle_illegal_parameter_error("gesvd", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "singular values failed to converge");
    }
}
template <>
void lapack_funcs<scalars::double_complex, double>::gesdd(
        const char* jobz, const integer m, const integer n,
        scalars::double_complex* A, const integer lda, double* s,
        scalars::double_complex* u, const integer ldu,
        scalars::double_complex* vt, const integer ldvt
)
{
    integer info = 0;
    auto* a = reinterpret_cast<complex64*>(A);
    auto* u_ = reinterpret_cast<complex64*>(u);
    auto* vt_ = reinterpret_cast<complex64*>(vt);

    reset_workspace();
    RPY_LAPACK_FUNC(zgesdd)
    (jobz, &m, &n, a, &lda, s, u_, &ldu, vt_, &ldvt, m_work.data(), &lwork,
     m_rwork.data(), m_iwork.data(), &info);
    RPY_CHECK(info == 0);
    resize_workspace(true, true);

    RPY_LAPACK_FUNC(zgesdd)
    (jobz, &m, &n, a, &lda, s, u_, &ldu, vt_, &ldvt, m_work.data(), &lwork,
     m_rwork.data(), m_iwork.data(), &info);
    if (info == -4) {
        RPY_THROW(std::invalid_argument, "matrix A contains a NaN value");
    } else if (info < 0) {
        handle_illegal_parameter_error("gesdd", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "singular values failed to converge");
    }
}
template <>
void lapack_funcs<scalars::double_complex, double>::gels(
        blas::BlasTranspose trans, const integer m, const integer n,
        const integer nrhs, scalars::double_complex* A, const integer lda,
        scalars::double_complex* B, const integer ldb
)
{
    integer info = 0;
    const auto* trans_ = reinterpret_cast<const char*>(&trans);
    auto* a = reinterpret_cast<complex64*>(A);
    auto* b = reinterpret_cast<complex64*>(B);

    reset_workspace();
    RPY_LAPACK_FUNC(zgels)
    (trans_, &m, &n, &nrhs, a, &lda, b, &ldb, m_work.data(), &lwork, &info);
    RPY_CHECK(info == 0);
    resize_workspace();

    RPY_LAPACK_FUNC(zgels)
    (trans_, &m, &n, &nrhs, a, &lda, b, &ldb, m_work.data(), &lwork, &info);
    if (info < 0) {
        handle_illegal_parameter_error("gels", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "matrix does not have full rank");
    }
}
template <>
void lapack_funcs<scalars::double_complex, double>::gelsy(
        const integer m, const integer n, const integer nrhs,
        scalars::double_complex* A, const integer lda,
        scalars::double_complex* B, const integer ldb, integer* jpvt,
        const double& rcond, integer& rank
)
{
    integer info = 0;
    auto* a = reinterpret_cast<complex64*>(A);
    auto* b = reinterpret_cast<complex64*>(B);
    auto* rcond_ = &rcond;

    reset_workspace();
    RPY_LAPACK_FUNC(zgelsy)
    (&m, &n, &nrhs, a, &lda, b, &ldb, jpvt, rcond_, &rank, m_work.data(),
     &lwork, m_rwork.data(), &info);
    RPY_CHECK(info == 0);
    resize_workspace(true);

    RPY_LAPACK_FUNC(zgelsy)
    (&m, &n, &nrhs, a, &lda, b, &ldb, jpvt, rcond_, &rank, m_work.data(),
     &lwork, m_rwork.data(), &info);
    if (info < 0) {
        handle_illegal_parameter_error("gelsy", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "matrix does not have full rank");
    }
}
template <>
void lapack_funcs<scalars::double_complex, double>::gelss(
        const integer m, const integer n, const integer nrhs,
        scalars::double_complex* A, const integer lda,
        scalars::double_complex* B, const integer ldb, double* s,
        const double& rcond, integer& rank
)
{
    integer info = 0;
    auto* a = reinterpret_cast<complex64*>(A);
    auto* b = reinterpret_cast<complex64*>(B);
    const auto* rcond_ = &rcond;

    reset_workspace();
    RPY_LAPACK_FUNC(zgelss)
    (&m, &n, &nrhs, a, &lda, b, &ldb, s, rcond_, &rank, m_work.data(), &lwork,
     m_rwork.data(), &info);
    RPY_CHECK(info == 0);
    resize_workspace(true);

    RPY_LAPACK_FUNC(zgelss)
    (&m, &n, &nrhs, a, &lda, b, &ldb, s, rcond_, &rank, m_work.data(), &lwork,
     m_rwork.data(), &info);
    if (info < 0) {
        handle_illegal_parameter_error("gelss", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "algorithm for computing svd failed");
    }
}
template <>
void lapack_funcs<scalars::double_complex, double>::gelsd(
        const integer m, const integer n, const integer nrhs,
        scalars::double_complex* A, const integer lda,
        scalars::double_complex* B, const integer ldb, double* s,
        const double& rcond, integer& rank
)
{
    integer info = 0;

    auto* a = reinterpret_cast<complex64*>(A);
    auto* b = reinterpret_cast<complex64*>(B);
    const auto* rcond_ = &rcond;

    reset_workspace();
    RPY_LAPACK_FUNC(zgelsd)
    (&m, &n, &nrhs, a, &lda, b, &ldb, s, rcond_, &rank, m_work.data(), &lwork,
     m_rwork.data(), m_iwork.data(), &info);
    RPY_CHECK(info == 0);
    resize_workspace(true, true);

    RPY_LAPACK_FUNC(zgelsd)
    (&m, &n, &nrhs, a, &lda, b, &ldb, s, rcond_, &rank, m_work.data(), &lwork,
     m_rwork.data(), m_iwork.data(), &info);
    if (info < 0) {
        handle_illegal_parameter_error("gelsd", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "algorithm for computing svd failed");
    }
}

}// namespace lapack

}// namespace rpy
