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

#include <roughpy/scalars/scalars_fwd.h>

#include <roughpy/scalars/owned_scalar_array.h>
#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_array.h>
#include <roughpy/scalars/scalar_blas.h>

#include <sstream>

#include "linear_algebra/blas.h"
#include "linear_algebra/lapack.h"



namespace rpy {
namespace scalars {

template <typename S, typename R>
class StandardLinearAlgebra : public BlasInterface,
                              blas::blas_funcs<S, R>,
                              lapack::lapack_funcs<S, R>
{

    using blas = blas::blas_funcs<S, R>;
    using lapack = lapack::lapack_funcs<S, R>;

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

public:

    using BlasInterface::BlasInterface;

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
        BlasTranspose transa = BlasTranspose::CblasNoTrans;

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
                blas::to_blas_layout(A.layout()), transa, m, n, alp,
                A.raw_cast<const scalar*>(), lda, x.raw_cast<const scalar*>(),
                incx, bet, y.raw_cast<scalar*>(), incy
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
            transa = BlasTranspose::CblasNoTrans;
            transb = (B.layout() == A.layout()) ? BlasTranspose::CblasNoTrans
                                                : BlasTranspose::CblasTrans;
        } else {
            type_check(C);
            RPY_CHECK(C.nrows() == m && C.ncols() == n);

            ldc = C.leading_dimension();
            //            layout = to_blas_layout(C.layout());

            transa = (A.layout() == C.layout()) ? BlasTranspose::CblasNoTrans
                                                : BlasTranspose::CblasTrans;

            transb = (B.layout() == C.layout()) ? BlasTranspose::CblasNoTrans
                                                : BlasTranspose::CblasTrans;
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

        const auto lda = A.leading_dimension();
        const auto ldb = B.leading_dimension();

        std::vector<integer> ipiv(n);

        lapack::gesv(
                blas::to_blas_layout(A.layout()),
                n, nrhs, A.raw_cast<scalar*>(), lda, ipiv.data(),
                B.raw_cast<scalar*>(), ldb
        );
    }
    EigenDecomposition syev(ScalarMatrix& A, bool eigenvectors) override
    {
        auto guard = lock();
        const char jobz = eigenvectors ? 'V' : 'N';
        auto uplo = BlasUpLo::CblasUpper;

        const auto n = A.nrows();
        const auto lda = A.leading_dimension();

        EigenDecomposition result;
        result.Lambda = ScalarMatrix(type(), n, 1, MatrixLayout::ColumnMajor);

        //        integer ldz = 1;
        //        scalar* vs = nullptr;
        //        if (eigenvectors) {
        //            result.U = ScalarMatrix(type(), n, n, A.layout());
        //            vs = result.U.raw_cast<scalar*>();
        //            ldz = result.U.leading_dimension();
        //        }

        lapack::syev(
                blas::to_blas_layout(A.layout()),
                &jobz, uplo, n, A.raw_cast<scalar*>(), lda,
                result.Lambda.raw_cast<scalar*>()
        );
        if (eigenvectors) { result.U = A; }
        return result;
    }
    EigenDecomposition geev(ScalarMatrix& A, bool eigenvectors) override
    {
        auto guard = lock();
        type_check(A);
        RPY_CHECK(!A.is_null());
        RPY_CHECK(A.ncols() == A.nrows());

        const auto layout = blas::to_blas_layout(A.layout());
        const auto n = A.nrows();
        const auto lda = A.leading_dimension();

        const auto jobvl = eigenvectors ? 'V' : 'N';
        const auto jobvr = 'N';
        LAPACK_S_SELECT2 select = nullptr;

        EigenDecomposition result;

        integer sdim = 0;
        integer ldvl = (eigenvectors) ? n : 1;

        /*
         * At the moment, only left eigenvalues are supported
         */
        integer ldvr = 1;

        /*
         * The eigenvalues of a real, non-symmetric matrix can be complex, so
         * write the real/imaginary parts into two temporary buffers.
         * Afterwards, we can choose whether the result vector has type float or
         * complex<float> depending on whether any of the eigenvalues are
         * complex.
         */
        std::vector<scalar> ev_real(n);
        std::vector<scalar> ev_imag(n);
        std::vector<scalar> vl;
        std::vector<scalar> vr;

        lapack::geev(
                blas::to_blas_layout(A.layout()),
                &jobvl, &jobvr, n, A.raw_cast<scalar*>(), lda,
                ev_real.data(), ev_imag.data(), vl.data(), ldvl, nullptr, ldvr
        );

        /*
         * Even if the input matrix was complex, we still check ev_imag to
         * avoid any funky logic here. Anyway this is O(n), and compared to
         * the linear algebra we have just done, not really consequential.
         * For complex scalar types, ev_imag is never referenced, so the
         * resulting complex_evs vector is empty.
         */
        std::vector<integer> complex_evs;
        complex_evs.reserve(n);
        integer i = 0;
        for (const auto& imag : ev_imag) {
            if (imag != scalar(0)) { complex_evs.push_back(i); }
            ++i;
        }

        if (!complex_evs.empty()) {
            result.Lambda = ScalarMatrix(type(), n, 1);
            ////// TODO: Implement complex number stuff
            if (eigenvectors) {
                result.U = ScalarMatrix(type(), n, n, A.layout());
            }

        } else {
            result.Lambda = ScalarMatrix(type(), n, 1);
            std::copy_n(ev_real.begin(), n, result.Lambda.raw_cast<scalar*>());
            if (eigenvectors) {
                result.U = ScalarMatrix(type(), n, n, A.layout());
                std::copy(vl.begin(), vl.end(), result.U.raw_cast<scalar*>());
            }
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

        lapack::gesvd(
                blas::to_blas_layout(A.layout()),
                &jobu, &jobvt, m, n, A.raw_cast<scalar*>(), lda,
                result.Sigma.raw_cast<scalar*>(), u, ldu, vt, ldvt
        );

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

        lapack::gesdd(
                blas::to_blas_layout(A.layout()),
                &jobz, m, n, A.raw_cast<scalar*>(), lda,
                result.Sigma.raw_cast<scalar*>(), u, ldu, vt, ldvt
        );

        return result;
    }
    void gels(ScalarMatrix& A, ScalarMatrix& b) override
    {
        auto guard = lock();
        type_check(A);
        type_check(b);
        RPY_CHECK(!A.is_null());
        RPY_CHECK(!b.is_null());

        const auto layout = blas::to_blas_layout(A.layout());
        const auto m = A.nrows();
        const auto n = A.ncols();
        const auto lda = A.leading_dimension();

        BlasTranspose trans;
        integer nrhs;
        integer ldb;

        if (b.layout() == MatrixLayout::RowMajor) {
            nrhs = b.nrows();
            ldb = b.ncols();
            trans = BlasTranspose::CblasTrans;
        } else {
            nrhs = b.ncols();
            ldb = b.nrows();
            trans = BlasTranspose::CblasNoTrans;
        }

        lapack::gels(
                blas::to_blas_layout(A.layout()),
                trans, m, n, nrhs, A.raw_cast<scalar*>(), lda,
                b.raw_cast<scalar*>(), ldb
        );
    }
    ScalarMatrix gelsy(ScalarMatrix& A, ScalarMatrix& b) override
    {
        auto guard = lock();
        type_check(A);
        type_check(b);
        RPY_CHECK(!A.is_null());
        RPY_CHECK(!b.is_null());

        const auto m = A.nrows();
        const auto n = A.ncols();
        const auto lda = A.leading_dimension();

        BlasTranspose trans;
        integer nrhs;
        integer ldb;

        if (b.layout() == MatrixLayout::RowMajor) {
            nrhs = b.nrows();
            ldb = b.ncols();
            trans = BlasTranspose::CblasTrans;
        } else {
            nrhs = b.ncols();
            ldb = b.nrows();
            trans = BlasTranspose::CblasNoTrans;
        }


        return {};
    }
    ScalarMatrix gelss(ScalarMatrix& A, ScalarMatrix& b) override;

    ScalarMatrix gelsd(ScalarMatrix& A, ScalarMatrix& b) override;
};

template <typename S, typename R>
ScalarMatrix
StandardLinearAlgebra<S, R>::gelss(ScalarMatrix& A, ScalarMatrix& b)
{
    auto guard = lock();
    type_check(A);
    type_check(b);
    RPY_CHECK(!A.is_null());
    RPY_CHECK(!b.is_null());

    const auto m = A.nrows();
    const auto n = A.ncols();
    const auto lda = A.leading_dimension();

    integer nrhs;
    integer ldb;
    BlasTranspose trans;

    if (b.layout() == MatrixLayout::RowMajor) {
        nrhs = b.nrows();
        ldb = b.ncols();
    } else {
        nrhs = b.ncols();
        ldb = b.nrows();
    }

    std::vector<real_scalar> singular_vals(std::min(m, n));
    std::vector<integer> pivots(n);
    real_scalar rcond = 0;
    integer rank = 0;

    lapack::gelss(
            blas::to_blas_layout(A.layout()),
            m,
            n,
            nrhs,
            A.raw_cast<scalar*>(),
            lda,
            b.raw_cast<scalar*>(),
            ldb,
            singular_vals.data(),
            rcond,
            rank
            );
    //    lapack::gelss(A.layout(), m, n, )
}
template <typename S, typename R>
ScalarMatrix
StandardLinearAlgebra<S, R>::gelsd(ScalarMatrix& A, ScalarMatrix& b)
{
    auto guard = lock();
    type_check(A);
    type_check(b);
    RPY_CHECK(!A.is_null());
    RPY_CHECK(!b.is_null());

    const auto m = A.nrows();
    const auto n = A.ncols();
    const auto lda = A.leading_dimension();

    integer nrhs;
    integer ldb;
    BlasTranspose trans;

    if (b.layout() == MatrixLayout::RowMajor) {
        nrhs = b.nrows();
        ldb = b.ncols();
    } else {
        nrhs = b.ncols();
        ldb = b.nrows();
    }

    std::vector<real_scalar> singular_vals(std::min(m, n));
    std::vector<integer> pivots(n);
    real_scalar rcond = 0;
    integer rank = 0;

    lapack::gelsd(
            blas::to_blas_layout(A.layout()),
            m,
            n,
            nrhs,
            A.raw_cast<scalar*>(),
            lda,
            b.raw_cast<scalar*>(),
            ldb,
            singular_vals.data(),
            rcond,
            rank
    );
    return {};
}

}// namespace scalars

}// namespace rpy
