// Copyright (c) 2023 Datasig Developers. All rights reserved.
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
// Created by user on 17/04/23.
//

#include "float_blas.h"
#include "scalar_blas_defs.h"

#include <roughpy/scalars/owned_scalar_array.h>
#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_array.h>
#include <roughpy/scalars/scalar_matrix.h>
#include <roughpy/scalars/scalar_pointer.h>
#include <roughpy/scalars/scalar_type.h>

using namespace rpy;
using namespace rpy::scalars;

OwnedScalarArray FloatBlas::vector_axpy(const ScalarArray& x, const Scalar& a,
                                        const ScalarArray& y)
{
    const auto* type = BlasInterface::type();
    RPY_CHECK(x.type() == type && y.type() == type);
    OwnedScalarArray result(type, y.size());
    type->convert_copy(result.ptr(), y, y.size());

    auto N = static_cast<blas::integer>(y.size());
    cblas_saxpy(N, scalar_cast<float>(a), x.raw_cast<const float*>(), 1,
                result.raw_cast<float*>(), 1);
    return result;
}
Scalar FloatBlas::dot_product(const ScalarArray& lhs, const ScalarArray& rhs)
{
    const auto* type = BlasInterface::type();

    RPY_CHECK(lhs.type() == type && rhs.type() == type);

    auto N = static_cast<blas::integer>(lhs.size());
    auto result = cblas_sdot(N, lhs.raw_cast<const float*>(), 1,
                             rhs.raw_cast<const float*>(), 1);
    return {type, result};
}
Scalar FloatBlas::L1Norm(const ScalarArray& vector)
{
    auto N = static_cast<blas::integer>(vector.size());
    auto result = cblas_sasum(N, vector.raw_cast<const float*>(), 1);
    return {type(), result};
}
Scalar FloatBlas::L2Norm(const ScalarArray& vector)
{
    RPY_CHECK(vector.type() == type());
    float result = 0.0;
    auto N = static_cast<blas::integer>(vector.size());
    result = cblas_snrm2(N, vector.raw_cast<const float*>(), 1);
    return {type(), result};
}
Scalar FloatBlas::LInfNorm(const ScalarArray& vector)
{
    RPY_CHECK(vector.type() == type());
    auto N = static_cast<blas::integer>(vector.size());
    const auto* ptr = vector.raw_cast<const float*>();
    auto idx = cblas_isamax(N, ptr, 1);
    auto result = ptr[idx];
    return {type(), result};
}
OwnedScalarArray FloatBlas::matrix_vector(const ScalarMatrix& matrix,
                                          const ScalarArray& vector)
{
    RPY_CHECK(matrix.type() == type() && vector.type() == type());

    auto M = static_cast<blas::integer>(matrix.nrows());
    auto N = static_cast<blas::integer>(matrix.ncols());

    if (N != static_cast<blas::integer>(vector.size())) {
        RPY_THROW(std::invalid_argument, "inner matrix dimensions must agree");
    }

    const auto layout = blas::to_blas_layout(matrix.layout());
    OwnedScalarArray result(type(), M);

    // TODO: fix lda/ldb values
    switch (matrix.storage()) {
        case MatrixStorage::FullMatrix:
            cblas_sgemv(layout, blas::Blas_NoTrans, M, N, 1.0F,
                        matrix.raw_cast<const float*>(), 1,
                        vector.raw_cast<const float*>(), 1, 0.0F,
                        result.raw_cast<float*>(), 1);
            break;
        case MatrixStorage::LowerTriangular:
        case MatrixStorage::UpperTriangular:
            RPY_CHECK(M == N);
            type()->convert_copy(result.ptr(), vector, vector.size());
            cblas_stpmv(layout, blas::to_blas_uplo(matrix.storage()),
                        blas::Blas_NoTrans, blas::Blas_DNoUnit, N,
                        matrix.raw_cast<const float*>(),
                        result.raw_cast<float*>(), 1);
            break;
        case MatrixStorage::Diagonal:
            RPY_CHECK(M == N);
            cblas_ssbmv(layout, blas::Blas_Lo, N, 1, 1.0F,
                        matrix.raw_cast<const float*>(), 1,
                        vector.raw_cast<const float*>(), 1, 0.0F,
                        result.raw_cast<float*>(), 1);
            break;
    }
    return result;
}

static constexpr MatrixLayout flip_layout(MatrixLayout layout) noexcept
{
    return layout == MatrixLayout::CStype ? MatrixLayout::FStype
                                          : MatrixLayout::CStype;
}

static void tfmm(ScalarMatrix& result, const ScalarMatrix& lhs,
                 blas::BlasUpLo lhs_uplo, const ScalarMatrix& rhs)
{
    {
        auto N = static_cast<blas::integer>(rhs.size());
        cblas_scopy(N, rhs.raw_cast<const float*>(), 1,
                    result.raw_cast<float*>(), 1);
        result.layout(rhs.layout());
    }
    auto C = static_cast<blas::integer>(result.ncols());

    auto layout = blas::to_blas_layout(lhs.layout());

    // If rhs is row-major then we need to adjust incx to stride correctly in
    // result.
    blas::integer incx = (result.layout() == MatrixLayout::FStype) ? 1 : C;

    auto* out_ptr = result.raw_cast<float*>();

    for (blas::integer i = 0; i < C; ++i) {
        cblas_stpmv(layout, lhs_uplo, blas::Blas_NoTrans, blas::Blas_DUnit, C,
                    lhs.raw_cast<const float*>(), out_ptr + i, incx);
    }
}
static void ftmm(ScalarMatrix& result, const ScalarMatrix& lhs,
                 const ScalarMatrix& rhs, blas::BlasUpLo rhs_uplo)
{
    // Use AT = (T'A')'

    ScalarMatrix right{lhs.nrows(), lhs.ncols(), ScalarArray(lhs),
                       lhs.storage(), flip_layout(lhs.layout())};

    ScalarMatrix left{rhs.nrows(), rhs.ncols(), ScalarArray(rhs), rhs.storage(),
                      flip_layout(rhs.layout())};

    tfmm(result, left,
         rhs_uplo == blas::Blas_Lo ? blas::Blas_Up : blas::Blas_Lo, right);
    result.layout(flip_layout(result.layout()));
}
static void dfmm(ScalarMatrix& result, const ScalarMatrix& lhs,
                 const ScalarMatrix& rhs)
{
    // lhs is diagonal, so the result of multiplying is just scaling each
    // corresponding column of rhs.
    RPY_DBG_ASSERT(lhs.ncols() == rhs.nrows());
    auto M = static_cast<blas::integer>(rhs.nrows());

    blas::integer ldb;
    if (rhs.layout() == MatrixLayout::CStype) {
        ldb = static_cast<blas::integer>(rhs.ncols());
    } else {
        ldb = 1;
    }

    result.layout(rhs.layout());

    auto* out_ptr = result.raw_cast<float*>();
    // LHS is packed, containing M = N values in a flat vector.
    const auto* lhs_ptr = lhs.raw_cast<const float*>();
    const auto* rhs_ptr = rhs.raw_cast<const float*>();

    for (blas::integer i = 0; i < ldb; ++i) {
        cblas_saxpy(M, lhs_ptr[i], rhs_ptr + i, ldb, out_ptr + i, ldb);
    }
}
static void fdmm(ScalarMatrix& result, const ScalarMatrix& lhs,
                 const ScalarMatrix& rhs)
{
    // Use the fact that AB = (B'A')'

    ScalarMatrix right{lhs.nrows(), lhs.ncols(), ScalarArray(lhs),
                       lhs.storage(), flip_layout(lhs.layout())};

    ScalarMatrix left{rhs.nrows(), rhs.ncols(), ScalarArray(rhs), rhs.storage(),
                      flip_layout(rhs.layout())};

    dfmm(result, left, right);
    result.layout(flip_layout(result.layout()));
}

static void ffmm(ScalarMatrix& result, const ScalarMatrix& lhs,
                 const ScalarMatrix& rhs)
{
    auto M = static_cast<blas::integer>(lhs.nrows());
    auto K = static_cast<blas::integer>(lhs.ncols());
    auto N = static_cast<blas::integer>(rhs.ncols());

    blas::BlasLayout layout;
    blas::BlasTranspose transa;
    blas::BlasTranspose transb;
    blas::integer lda;
    blas::integer ldb;
    blas::integer ldc;

    if (lhs.layout() == rhs.layout()) {
        layout = blas::to_blas_layout(lhs.layout());
        result.layout(lhs.layout());
        transa = blas::Blas_NoTrans;
        transb = blas::Blas_NoTrans;
        lda = (layout == blas::Blas_ColMajor) ? M : K;
        ldb = (layout == blas::Blas_ColMajor) ? K : N;
        ldc = (layout == blas::Blas_ColMajor) ? M : N;
    } else if (lhs.layout() == MatrixLayout::CStype) {
        layout = blas::Blas_RowMajor;
        result.layout(MatrixLayout::CStype);
        transa = blas::Blas_NoTrans;
        transb = blas::Blas_Trans;
        lda = K;
        ldb = K;
        ldc = N;
    } else {
        layout = blas::Blas_ColMajor;
        result.layout(MatrixLayout::FStype);
        transa = blas::Blas_NoTrans;
        transb = blas::Blas_Trans;
        lda = M;
        ldb = N;
        ldc = M;
    }

    cblas_sgemm(layout, transa, transb, M, N, K, 1.0F,
                lhs.raw_cast<const float*>(), lda, rhs.raw_cast<const float*>(),
                ldb, 0.0F, result.raw_cast<float*>(), ldc);
}

static void ttmm(ScalarMatrix& result, const ScalarMatrix& lhs,
                 blas::BlasUpLo lhs_uplo, const ScalarMatrix& rhs,
                 blas::BlasUpLo rhs_uplo)
{
    /*
     * If lhs and rhs are both upper (lower, respectively) triangular then
     * result will also be upper (lower) triangular. Otherwise, the result is a
     * full matrix.
     */
}

static void dtmm(ScalarMatrix& result, const ScalarMatrix& lhs,
                 const ScalarMatrix& rhs, blas::BlasUpLo rhs_uplo)
{
    // There is no BLAS routine for this so we're going to have to do it
    // ourselves.
}

static void tdmm(ScalarMatrix& result, const ScalarMatrix& lhs,
                 blas::BlasUpLo lhs_uplo, const ScalarMatrix& rhs)
{
    // Uss the fact that TD = (D'T')' = (DT')'
    const auto old_storage = result.storage();
    const auto old_layout = result.layout();

    ScalarMatrix right{lhs.nrows(), lhs.ncols(), ScalarArray(lhs),
                       lhs.storage() == MatrixStorage::UpperTriangular
                               ? MatrixStorage::LowerTriangular
                               : MatrixStorage::UpperTriangular,
                       flip_layout(lhs.layout())};

    result.layout(right.layout());
    result.storage(right.storage());

    dtmm(result, rhs, right,
         lhs_uplo == blas::Blas_Up ? blas::Blas_Lo : blas::Blas_Up);
    result.layout(old_layout);
    result.storage(old_storage);
}

static void ddmm(ScalarMatrix& result, const ScalarMatrix& lhs,
                 const ScalarMatrix& rhs)
{
    RPY_DBG_ASSERT(result.storage() == MatrixStorage::Diagonal);

    auto* out_ptr = result.raw_cast<float*>();
    const auto* lhs_ptr = lhs.raw_cast<const float*>();
    const auto* rhs_ptr = rhs.raw_cast<const float*>();

    // This is a silly simple case. just do the multiplication manually.
    for (deg_t i = 0; i < lhs.ncols(); ++i) {
        out_ptr[i] = lhs_ptr[i] * rhs_ptr[i];
    }
}

static void full_matrix_matrix(ScalarMatrix& result, const ScalarMatrix& lhs,
                               const ScalarMatrix& rhs)
{

    switch (rhs.storage()) {
        case MatrixStorage::FullMatrix: ffmm(result, lhs, rhs); break;
        case MatrixStorage::LowerTriangular:
            ftmm(result, lhs, rhs, blas::Blas_Lo);
            break;
        case MatrixStorage::UpperTriangular:
            ftmm(result, lhs, rhs, blas::Blas_Up);
            break;
        case MatrixStorage::Diagonal: fdmm(result, lhs, rhs); break;
        default:
            RPY_THROW(std::runtime_error,"matrix-matrix multiplications of these "
                                     "formats is not currently supported");
    }
}

static void triangular_matrix_matrix(ScalarMatrix& result,
                                     const ScalarMatrix& lhs,
                                     blas::BlasUpLo lhs_uplo,
                                     const ScalarMatrix& rhs)
{

    switch (rhs.storage()) {
        case MatrixStorage::FullMatrix: tfmm(result, lhs, lhs_uplo, rhs); break;
        case MatrixStorage::UpperTriangular:
            ttmm(result, lhs, lhs_uplo, rhs, blas::Blas_Up);
            break;
        case MatrixStorage::LowerTriangular:
            ttmm(result, lhs, lhs_uplo, rhs, blas::Blas_Lo);
            break;
        case MatrixStorage::Diagonal: tdmm(result, lhs, lhs_uplo, rhs); break;
        default:
            RPY_THROW(std::runtime_error,"matrix-matrix multiplications of these "
                                     "formats is currently unsupported");
    }
}
static void diagonal_matrix_matrix(ScalarMatrix& result,
                                   const ScalarMatrix& lhs,
                                   const ScalarMatrix& rhs)
{
    switch (rhs.storage()) {
        case MatrixStorage::FullMatrix: dfmm(result, lhs, rhs); break;
        case MatrixStorage::LowerTriangular:
            dtmm(result, lhs, rhs, blas::Blas_Lo);
            break;
        case MatrixStorage::UpperTriangular:
            dtmm(result, lhs, rhs, blas::Blas_Up);
            break;
        case MatrixStorage::Diagonal: ddmm(result, lhs, rhs); break;
        default:
            RPY_THROW(std::runtime_error,"matrix-matrix multiplications of these "
                                     "formats is currently unsupported");
    }
}

ScalarMatrix FloatBlas::matrix_matrix(const ScalarMatrix& lhs,
                                      const ScalarMatrix& rhs)
{
    RPY_DBG_ASSERT(lhs.type() == rhs.type() && lhs.type() == type());

    if (lhs.ncols() != rhs.nrows()) {
        RPY_THROW(std::invalid_argument, "inner matrix dimensions must agree");
    };

    ScalarMatrix result(type(), lhs.nrows(), rhs.ncols());

    switch (lhs.storage()) {
        case MatrixStorage::FullMatrix:
            full_matrix_matrix(result, lhs, rhs);
            break;
        case MatrixStorage::UpperTriangular:
            triangular_matrix_matrix(result, lhs, blas::Blas_Up, rhs);
            break;
        case MatrixStorage::LowerTriangular:
            triangular_matrix_matrix(result, lhs, blas::Blas_Lo, rhs);
            break;
        case MatrixStorage::Diagonal:
            diagonal_matrix_matrix(result, lhs, rhs);
            break;
    }

    return result;
}
ScalarMatrix FloatBlas::solve_linear_system(const ScalarMatrix& coeff_matrix,
                                            const ScalarMatrix& target_matrix)
{

    if (coeff_matrix.nrows() != target_matrix.nrows()) {
        RPY_THROW(std::invalid_argument, "incompatible matrix dimensions");
    }

    if (coeff_matrix.nrows() < coeff_matrix.ncols()) {
        RPY_THROW(std::invalid_argument,
                "system is over-determined, used least squares instead");
    }
    if (coeff_matrix.nrows() > coeff_matrix.ncols()) {
        RPY_THROW(std::invalid_argument, "system is under-determined, no solution");
    }

    ScalarMatrix result = target_matrix.to_full();

    //    switch (coeff_matrix.storage()) {
    //        case MatrixStorage::FullMatrix:
    //
    //            break;
    //    }
    //
    return result;
}
OwnedScalarArray FloatBlas::lls_qr(const ScalarMatrix& matrix,
                                   const ScalarArray& target)
{

    switch (matrix.storage()) {
        case MatrixStorage::FullMatrix: break;
        case MatrixStorage::UpperTriangular: break;
        case MatrixStorage::LowerTriangular: break;
        case MatrixStorage::Diagonal: break;
    }
    return {};
}
OwnedScalarArray FloatBlas::lls_orth(const ScalarMatrix& matrix,
                                     const ScalarArray& target)
{
    return BlasInterface::lls_orth(matrix, target);
}
OwnedScalarArray FloatBlas::lls_svd(const ScalarMatrix& matrix,
                                    const ScalarArray& target)
{
    return BlasInterface::lls_svd(matrix, target);
}
OwnedScalarArray FloatBlas::lls_dcsvd(const ScalarMatrix& matrix,
                                      const ScalarArray& target)
{
    return BlasInterface::lls_dcsvd(matrix, target);
}
OwnedScalarArray FloatBlas::lse_grq(const ScalarMatrix& A,
                                    const ScalarMatrix& B, const ScalarArray& c,
                                    const ScalarArray& d)
{
    return BlasInterface::lse_grq(A, B, c, d);
}
ScalarMatrix FloatBlas::glm_GQR(const ScalarMatrix& A, const ScalarMatrix& B,
                                const ScalarArray& d)
{
    return BlasInterface::glm_GQR(A, B, d);
}
EigenDecomposition FloatBlas::eigen_decomposition(const ScalarMatrix& matrix)
{
    return BlasInterface::eigen_decomposition(matrix);
}
SingularValueDecomposition FloatBlas::svd(const ScalarMatrix& matrix)
{
    return BlasInterface::svd(matrix);
}
void FloatBlas::transpose(ScalarMatrix& matrix) const {}
