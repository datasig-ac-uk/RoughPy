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
// Created by user on 17/04/23.
//

#include "float_blas.h"


using namespace rpy;
using namespace rpy::scalars;

namespace rpy { namespace scalars {
template class StandardLinearAlgebra<float, float>;
}}

//
//void FloatBlas::check_and_report_errors(matrix_dim_t info) const
//{
//    if (info == 0) { return; }
//
//    RPY_THROW(std::runtime_error, "BLAS/LAPACK encountered an error");
//}
//
//OwnedScalarArray FloatBlas::vector_axpy(
//        const ScalarArray& x, const Scalar& a, const ScalarArray& y
//)
//{
//    auto guard = lock();
//    const auto* type = BlasInterface::type();
//    RPY_CHECK(x.type() == type && y.type() == type);
//    OwnedScalarArray result(type, y.size());
//    type->convert_copy(result, y, y.size());
//
//    auto N = static_cast<blas::integer>(y.size());
//    cblas_saxpy(
//            N, scalar_cast<float>(a), x.raw_cast<const float*>(), 1,
//            result.raw_cast<float*>(), 1
//    );
//    return result;
//}
//Scalar FloatBlas::dot_product(const ScalarArray& lhs, const ScalarArray& rhs)
//{
//    auto guard = lock();
//    const auto* type = BlasInterface::type();
//
//    RPY_CHECK(lhs.type() == type && rhs.type() == type);
//
//    auto N = static_cast<blas::integer>(lhs.size());
//    auto result = cblas_sdot(
//            N, lhs.raw_cast<const float*>(), 1, rhs.raw_cast<const float*>(), 1
//    );
//    return {type, result};
//}
//Scalar FloatBlas::L1Norm(const ScalarArray& vector)
//{
//    auto guard = lock();
//    auto N = static_cast<blas::integer>(vector.size());
//    auto result = cblas_sasum(N, vector.raw_cast<const float*>(), 1);
//    return {type(), result};
//}
//Scalar FloatBlas::L2Norm(const ScalarArray& vector)
//{
//    RPY_CHECK(vector.type() == type());
//    float result = 0.0;
//    auto N = static_cast<blas::integer>(vector.size());
//    result = cblas_snrm2(N, vector.raw_cast<const float*>(), 1);
//    return {type(), result};
//}
//Scalar FloatBlas::LInfNorm(const ScalarArray& vector)
//{
//    RPY_CHECK(vector.type() == type());
//    auto N = static_cast<blas::integer>(vector.size());
//    const auto* ptr = vector.raw_cast<const float*>();
//    auto idx = cblas_isamax(N, ptr, 1);
//    auto result = ptr[idx];
//    return {type(), result};
//}
//
//void FloatBlas::transpose(ScalarMatrix& matrix) const {}
//
//void FloatBlas::gemv(
//        ScalarMatrix& y, const ScalarMatrix& A, const ScalarMatrix& x,
//        const Scalar& alpha, const Scalar& beta
//)
//{
//    auto guard = lock();
//    type_check(A);
//    type_check(x);
//    const float alp = scalar_cast<float>(alpha);
//    const float bet = scalar_cast<float>(beta);
//
//    blas::integer m = A.nrows();
//    blas::integer n = A.ncols();
//    blas::integer lda = A.leading_dimension();
//
//    blas::integer incx;
//    blas::integer n_eqns;
//    if (x.layout() == MatrixLayout::RowMajor) {
//        incx = x.ncols();
//        n_eqns = x.nrows();
//    } else {
//        incx = x.nrows();
//        n_eqns = x.ncols();
//    }
//
//    /*
//     * We're assuming that the vectors are stored contiguously so that incx
//     * is simply the number of columns/rows depending on whether it is row
//     * major or column major.
//     */
//    matrix_product_check(n, incx);
//
//    /*
//     * For now, we shall assume that we don't want to transpose the matrix A.
//     * In the future we might want to consider transposing based on whether
//     * the matrix is in row major or column major format.
//     */
//    blas::BlasTranspose transa = blas::Blas_NoTrans;
//
//    blas::BlasLayout layout = A.layout() == MatrixLayout::RowMajor
//            ? blas::Blas_RowMajor
//            : blas::Blas_ColMajor;
//
//    blas::integer incy;
//    if (y.type() == nullptr || y.is_null()) {
//        /*
//         * Type is null, so it hasn't been
//         * initialized yet. Allocate a new empty matrix of the correct size.
//         */
//        y = ScalarMatrix(type(), m, n_eqns, MatrixLayout::ColumnMajor);
//        incy = m;
//    } else {
//        type_check(y);
//        blas::integer n_eqs_check;
//        if (y.layout() == MatrixLayout::RowMajor) {
//            incy = y.ncols();
//            n_eqs_check = y.nrows();
//        } else {
//            incy = y.nrows();
//            n_eqs_check = y.ncols();
//        }
//
//        RPY_CHECK(incy == incx && n_eqs_check == n_eqns);
//    }
//
//    cblas_sgemv(
//            layout, transa, m, n, alp, A.raw_cast<const float*>(), lda,
//            x.raw_cast<const float*>(), incx, bet, y.raw_cast<float*>(), incy
//    );
//}
//void FloatBlas::gemm(
//        ScalarMatrix& C, const ScalarMatrix& A, const ScalarMatrix& B,
//        const Scalar& alpha, const Scalar& beta
//)
//{
//    auto guard = lock();
//    type_check(A);
//    type_check(B);
//    const auto alp = scalar_cast<float>(alpha);
//    const auto bet = scalar_cast<float>(beta);
//
//    blas::integer m = A.nrows();
//    blas::integer n = B.ncols();
//    blas::integer k = A.ncols();
//
//    const auto lda = A.leading_dimension();
//    const auto ldb = B.leading_dimension();
//    blas::integer ldc;// initialized below
//
//    blas::BlasLayout layout;
//    blas::BlasTranspose transa;
//    blas::BlasTranspose transb;
//
//    /*
//     * If C is initialized, we use C.layout() to determine whether A and B
//     * need to be transposed or not. If C is not initialized, we use A.layout()
//     * to set the layout of C and determine whether we should transpose B.
//     */
//    if (C.type() == nullptr || C.is_null()) {
//        // Allocate a new result matrix with dimensions m by n
//        C = ScalarMatrix(type(), m, n, A.layout());
//        ldc = C.leading_dimension();
//        layout = blas::to_blas_layout(A.layout());
//        transa = blas::Blas_NoTrans;
//        transb = (B.layout() == A.layout()) ? blas::Blas_NoTrans
//                                            : blas::Blas_Trans;
//    } else {
//        type_check(C);
//        RPY_CHECK(C.nrows() == m && C.ncols() == n);
//
//        ldc = C.leading_dimension();
//        layout = blas::to_blas_layout(C.layout());
//
//        transa = (A.layout() == C.layout()) ? blas::Blas_NoTrans
//                                            : blas::Blas_Trans;
//
//        transb = (B.layout() == C.layout()) ? blas::Blas_NoTrans
//                                            : blas::Blas_Trans;
//    }
//
//    cblas_sgemm(
//            layout, transa, transb, m, n, k, alp, A.raw_cast<const float*>(),
//            lda, B.raw_cast<const float*>(), ldb, bet, C.raw_cast<float*>(), ldc
//    );
//}
//void FloatBlas::gesv(ScalarMatrix& A, ScalarMatrix& B)
//{
//    auto guard = lock();
//    // Solving A*X = B, solution written to B
//    type_check(A);
//    type_check(B);
//
//    blas::integer n = A.nrows();
//    RPY_CHECK(A.ncols() == n);
//    blas::integer nrhs = B.ncols();
//    matrix_product_check(n, B.nrows());
//
//    const auto layout = blas::to_blas_layout(A.layout());
//    const auto lda = A.leading_dimension();
//    const auto ldb = B.leading_dimension();
//    const auto ldx = ldb;// Assuming X and B have the same shape
//
//    LAPACKE_sgesv(
//            layout, n, nrhs, A.raw_cast<float*>(), B.raw_cast<float*>(), lda,
//            ldb, ldx
//    );
//}
//EigenDecomposition FloatBlas::syev(ScalarMatrix& A, bool eigenvectors)
//{
//    auto guard = lock();
//    const auto layout = blas::to_blas_layout(A.layout());
//    const char jobz = eigenvectors ? 'V' : 'N';
//    const char range = 'A';
//    const char uplo = 'U';// ?
//
//    const auto n = A.nrows();
//    const auto lda = A.leading_dimension();
//    const float* vl = nullptr;
//    const float* vu = nullptr;
//    const blas::integer* il = nullptr;
//    const blas::integer* iu = nullptr;
//
//    const float abstol = 0.0;// will be replaced by n*eps*||A||.
//    blas::integer m = 0;
//
//    EigenDecomposition result;
//    result.Lambda = ScalarMatrix(type(), n, 1, MatrixLayout::ColumnMajor);
//
//    blas::integer ldz = 1;
//    float* vs = nullptr;
//    if (eigenvectors) {
//        result.U = ScalarMatrix(type(), n, n, A.layout());
//        vs = result.U.raw_cast<float*>();
//        ldz = result.U.leading_dimension();
//    }
//
//    std::vector<blas::integer> isuppz(2 * n);
//
//    LAPACKE_ssyevr(
//            layout, jobz, range, uplo, n, A.raw_cast<float*>(), lda, vl, vu, il,
//            iu, abstol, &m, result.Lambda.raw_cast<float*>(), vs, ldz,
//            isuppz.data()
//    );
//}
//
//EigenDecomposition FloatBlas::geev(ScalarMatrix& A, bool eigenvectors)
//{
//    auto guard = lock();
//    type_check(A);
//    RPY_CHECK(!A.is_null());
//    RPY_CHECK(A.ncols() == A.nrows());
//
//    const auto layout = blas::to_blas_layout(A.layout());
//    const auto n = A.nrows();
//    const auto lda = A.leading_dimension();
//
//    const auto jobvs = eigenvectors ? 'V' : 'N';
//    const auto sort = 'N';
//    LAPACK_S_SELECT2 select = nullptr;
//
//    EigenDecomposition result;
//
//    blas::integer sdim = 0;
//    float* vs = nullptr;
//    blas::integer ldvs = 1;
//    if (eigenvectors) {
//        result.U = ScalarMatrix(type(), n, n, A.layout());
//        ldvs = n;
//        vs = result.U.raw_cast<float*>();
//    }
//
//    /*
//     * The eigenvalues of a real, non-symmetric matrix can be complex, so
//     * write the real/imaginary parts into two temporary buffers. Afterwards,
//     * we can choose whether the result vector has type float or
//     * complex<float> depending on whether any of the eigenvalues are complex.
//     */
//    std::vector<float> ev_real(n);
//    std::vector<float> ev_imag(n);
//
//    LAPACKE_sgees(
//            layout, jobvs, sort, select, n, A.raw_cast<float*>(), lda, &sdim,
//            ev_real.data(), ev_imag.data(), vs, ldvs
//    );
//
//    bool any_complex = std::any_of(ev_imag.begin(), ev_imag.end(), [](float v) {
//        return v != 0.0f;
//    });
//
//    if (any_complex) {
//        result.Lambda = ScalarMatrix(type(), n, 1);
//        ////// TODO: Implement complex number stuff
//    } else {
//        result.Lambda = ScalarMatrix(type(), n, 1);
//        std::copy_n(ev_real.begin(), n, result.Lambda.raw_cast<float*>());
//    }
//
//    return result;
//}
//SingularValueDecomposition
//FloatBlas::gesvd(ScalarMatrix& A, bool return_U, bool return_VT)
//{
//    auto guard = lock();
//    type_check(A);
//    RPY_CHECK(!A.is_null());
//
//    const auto layout = blas::to_blas_layout(A.layout());
//    const auto m = A.nrows();
//    const auto n = A.ncols();
//    const auto lda = A.leading_dimension();
//
//    blas::integer ldu = 1;
//    blas::integer ldvt = 1;
//    char jobu = 'N';
//    char jobvt = 'N';
//
//    SingularValueDecomposition result;
//    result.Sigma = ScalarMatrix(type(), n, 1, MatrixLayout::ColumnMajor);
//
//    float* u = nullptr;
//    float* vt = nullptr;
//
//    if (return_U) {
//        jobu = 'A';
//        result.U = ScalarMatrix(type(), m, m);
//        ldu = m;
//        u = result.U.raw_cast<float*>();
//    }
//
//    if (return_VT) {
//        jobvt = 'A';
//        result.VHermitian = ScalarMatrix(type(), n, n);
//        ldvt = n;
//        vt = result.VHermitian.raw_cast<float*>();
//    }
//
//    m_workspace.clear();
//    m_workspace.resize(std::min(m, n) - 2);
//
//    auto info = LAPACKE_sgesvd(
//            layout, jobu, jobvt, m, n, A.raw_cast<float*>(), lda,
//            result.Sigma.raw_cast<float*>(), u, ldu, vt, ldvt,
//            m_workspace.data()
//    );
//
//    // Handle errors that might have happened in LAPACK
//    check_and_report_errors(info);
//
//    return result;
//}
//SingularValueDecomposition
//FloatBlas::gesdd(ScalarMatrix& A, bool return_U, bool return_VT)
//{
//    auto guard = lock();
//    type_check(A);
//    RPY_CHECK(!A.is_null());
//
//    const auto layout = blas::to_blas_layout(A.layout());
//    const auto m = A.nrows();
//    const auto n = A.ncols();
//    const auto lda = A.leading_dimension();
//
//    blas::integer ldu = 1;
//    blas::integer ldvt = 1;
//    char jobz = 'N';
//
//    SingularValueDecomposition result;
//    result.Sigma = ScalarMatrix(type(), n, 1, MatrixLayout::ColumnMajor);
//
//    float* u = nullptr;
//    float* vt = nullptr;
//
//    /*
//     * Unlike sgesvd, we can either get both U and VT or neither. We maintain
//     * the same interface though, but just populate both if either are
//     * requested.
//     */
//    if (return_U || return_VT) {
//        jobz = 'A';
//        result.U = ScalarMatrix(type(), m, m);
//        ldu = m;
//        u = result.U.raw_cast<float*>();
//        result.VHermitian = ScalarMatrix(type(), n, n);
//        ldvt = n;
//        vt = result.VHermitian.raw_cast<float*>();
//    }
//
//    auto info = LAPACKE_sgesdd(
//            layout, jobz, m, n, A.raw_cast<float*>(), lda,
//            result.Sigma.raw_cast<float*>(), u, ldu, vt, ldvt
//    );
//
//    // Handle errors that might have happened in LAPACK
//    check_and_report_errors(info);
//
//    return result;
//}
//void FloatBlas::gels(ScalarMatrix& A, ScalarMatrix& b)
//{
//    auto guard = lock();
//    type_check(A);
//    type_check(b);
//    RPY_CHECK(!A.is_null());
//    RPY_CHECK(!b.is_null());
//
//    const auto layout = blas::to_blas_layout(A.layout());
//    const auto m = A.nrows();
//    const auto n = A.ncols();
//    const auto lda = A.leading_dimension();
//
//    /*
//     * For now we assume that we don't need to transpose A for this
//     * calculation. In the future, we might want to consider changing this to
//     * account for difference between matrix layouts.
//     */
//    blas::BlasTranspose trans = blas::Blas_NoTrans;
//    blas::integer nrhs;
//    blas::integer ldb;
//
//    if (b.layout() == MatrixLayout::RowMajor) {
//        nrhs = b.nrows();
//        ldb = b.ncols();
//    } else {
//        nrhs = b.ncols();
//        ldb = b.nrows();
//    }
//
//    auto info = LAPACKE_sgels(
//            layout, trans, m, n, nrhs, A.raw_cast<float*>(), lda,
//            b.raw_cast<float*>(), ldb
//    );
//
//    check_and_report_errors(info);
//}
//ScalarMatrix FloatBlas::gelsy(ScalarMatrix& A, ScalarMatrix& b)
//{
//    auto guard = lock();
//    type_check(A);
//    type_check(b);
//    RPY_CHECK(!A.is_null());
//    RPY_CHECK(!b.is_null());
//
//    const auto layout = blas::to_blas_layout(A.layout());
//    const auto m = A.nrows();
//    const auto n = A.ncols();
//    const auto lda = A.leading_dimension();
//
//    /*
//     * For now we assume that we don't need to transpose A for this
//     * calculation. In the future, we might want to consider changing this to
//     * account for difference between matrix layouts.
//     */
//    blas::BlasTranspose trans = blas::Blas_NoTrans;
//    blas::integer nrhs;
//    blas::integer ldb;
//
//    if (b.layout() == MatrixLayout::RowMajor) {
//        nrhs = b.nrows();
//        ldb = b.ncols();
//    } else {
//        nrhs = b.ncols();
//        ldb = b.nrows();
//    }
//
//    std::vector<blas::integer> pivots(n);
//    // rcond on 0 forces use of machine precision.
//    float rcond = 0.0f;
//    blas::integer rank = 0;
//
//    auto info = LAPACKE_sgelsy(
//            layout, m, n, nrhs, A.raw_cast<float*>(), lda, b.raw_cast<float*>(),
//            ldb, pivots.data(), rcond, &rank
//    );
//
//    check_and_report_errors(info);
//}
//ScalarMatrix FloatBlas::gelss(ScalarMatrix& A, ScalarMatrix& b)
//{
//    auto guard = lock();
//    type_check(A);
//    type_check(b);
//    RPY_CHECK(!A.is_null());
//    RPY_CHECK(!b.is_null());
//
//    const auto layout = blas::to_blas_layout(A.layout());
//    const auto m = A.nrows();
//    const auto n = A.ncols();
//    const auto lda = A.leading_dimension();
//
//    /*
//     * For now we assume that we don't need to transpose A for this
//     * calculation. In the future, we might want to consider changing this to
//     * account for difference between matrix layouts.
//     */
//    blas::BlasTranspose trans = blas::Blas_NoTrans;
//    blas::integer nrhs;
//    blas::integer ldb;
//
//    if (b.layout() == MatrixLayout::RowMajor) {
//        nrhs = b.nrows();
//        ldb = b.ncols();
//    } else {
//        nrhs = b.ncols();
//        ldb = b.nrows();
//    }
//
//    std::vector<blas::integer> pivots(n);
//    // rcond on 0 forces use of machine precision.
//    float rcond = 0.0f;
//    blas::integer rank = 0;
//
//    auto info = LAPACKE_sgelss(
//            layout, m, n, nrhs, A.raw_cast<float*>(), lda, b.raw_cast<float*>(),
//            ldb, pivots.data(), rcond, &rank
//    );
//
//    check_and_report_errors(info);
//}
//ScalarMatrix FloatBlas::gelsd(ScalarMatrix& A, ScalarMatrix& b)
//{
//    auto guard = lock();
//    type_check(A);
//    type_check(b);
//    RPY_CHECK(!A.is_null());
//    RPY_CHECK(!b.is_null());
//
//    const auto layout = blas::to_blas_layout(A.layout());
//    const auto m = A.nrows();
//    const auto n = A.ncols();
//    const auto lda = A.leading_dimension();
//
//    /*
//     * For now we assume that we don't need to transpose A for this
//     * calculation. In the future, we might want to consider changing this to
//     * account for difference between matrix layouts.
//     */
//    blas::BlasTranspose trans = blas::Blas_NoTrans;
//    blas::integer nrhs;
//    blas::integer ldb;
//
//    if (b.layout() == MatrixLayout::RowMajor) {
//        nrhs = b.nrows();
//        ldb = b.ncols();
//    } else {
//        nrhs = b.ncols();
//        ldb = b.nrows();
//    }
//
//    std::vector<blas::integer> pivots(n);
//    // rcond on 0 forces use of machine precision.
//    float rcond = 0.0f;
//    blas::integer rank = 0;
//
//    auto info = LAPACKE_sgelsd(
//            layout, m, n, nrhs, A.raw_cast<float*>(), lda, b.raw_cast<float*>(),
//            ldb, pivots.data(), rcond, &rank
//    );
//
//    check_and_report_errors(info);
//}
