// Copyright (c) 2023 Datasig Developers. All rights reserved.
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
#include "scalar_blas_impl.h"

#include "scalar.h"
#include "scalar_array.h"
#include "scalar_matrix.h"
#include "scalar_pointer.h"
#include "scalar_type.h"

using namespace rpy;
using namespace rpy::scalars;

ScalarArray FloatBlas::vector_axpy(const ScalarArray &x, const Scalar &a, const ScalarArray &y) {
    const auto *type = BlasInterface::type();
    assert(x.type() == type && y.type() == type);
    ScalarArray result(type->allocate(y.size()), y.size());
    type->convert_copy(result.ptr(), y, y.size());

    auto N = static_cast<blas::integer>(y.size());
    cblas_saxpy(N, scalar_cast<float>(a), x.raw_cast<const float *>(), 1, result.raw_cast<float *>(), 1);
    return result;
}
Scalar FloatBlas::dot_product(const ScalarArray &lhs, const ScalarArray &rhs) {
    const auto *type = BlasInterface::type();

    assert(lhs.type() == type && rhs.type() == type);

    auto N = static_cast<blas::integer>(lhs.size());
    auto result = cblas_sdot(N, lhs.raw_cast<const float *>(), 1,
                             rhs.raw_cast<const float *>(), 1);
    return {type, result};
}
Scalar FloatBlas::L1Norm(const ScalarArray &vector) {
    auto N = static_cast<blas::integer>(vector.size());
    auto result = cblas_sasum(N, vector.raw_cast<const float *>(), 1);
    return {type(), result};
}
Scalar FloatBlas::L2Norm(const ScalarArray &vector) {
    assert(vector.type() == type());
    float result = 0.0;
    auto N = static_cast<blas::integer>(vector.size());
    result = cblas_snrm2(N, vector.raw_cast<const float *>(), 1);
    return {type(), result};
}
Scalar FloatBlas::LInfNorm(const ScalarArray &vector) {
    assert(vector.type() == type());
    auto N = static_cast<blas::integer>(vector.size());
    const auto *ptr = vector.raw_cast<const float *>();
    auto idx = cblas_isamax(N, ptr, 1);
    auto result = ptr[idx];
    return {type(), result};
}
ScalarArray FloatBlas::matrix_vector(const ScalarMatrix &matrix, const ScalarArray &vector) {
    assert(matrix.type() == type() && vector.type() == type());

    auto M = static_cast<blas::integer>(matrix.nrows());
    auto N = static_cast<blas::integer>(matrix.ncols());

    if (N != static_cast<blas::integer>(vector.size())) {
        throw std::invalid_argument("inner matrix dimensions must agree");
    }

    const auto layout = blas::to_blas_layout(matrix.layout());
    ScalarArray result(type()->allocate(M), M);

    switch (matrix.storage()) {
        case MatrixStorage::FullMatrix:
            cblas_sgemv(layout,
                        blas::Blas_NoTrans,
                        M,
                        N,
                        1.0F,
                        matrix.raw_cast<const float *>(),
                        1,
                        vector.raw_cast<const float *>(),
                        1,
                        0.0F,
                        result.raw_cast<float *>(),
                        1);
            break;
        case MatrixStorage::LowerTriangular:
        case MatrixStorage::UpperTriangular:
            assert(M == N);
            type()->convert_copy(result.ptr(), vector, vector.size());
            cblas_stpmv(layout,
                        blas::to_blas_uplo(matrix.storage()),
                        blas::Blas_NoTrans,
                        blas::Blas_DNoUnit,
                        N,
                        matrix.raw_cast<const float *>(),
                        result.raw_cast<float *>(),
                        1);
            break;
        case MatrixStorage::Diagonal:
            assert(M == N);
            cblas_ssbmv(layout,
                        blas::Blas_Lo,
                        N,
                        1,
                        1.0F,
                        matrix.raw_cast<const float *>(), 1,
                        vector.raw_cast<const float *>(), 1,
                        0.0F,
                        result.raw_cast<float *>(),
                        1);
            break;
    }
    return result;
}
static void full_matrix_matrix(ScalarMatrix &result, const ScalarMatrix &lhs, const ScalarMatrix &rhs) {
    auto M = static_cast<blas::integer>(lhs.nrows());
    auto K = static_cast<blas::integer>(lhs.ncols());
    auto N = static_cast<blas::integer>(rhs.ncols());

    blas::BlasLayout layout;
    blas::BlasTranspose transa;
    blas::BlasTranspose transb;
    blas::integer lda, ldb, ldc;

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


    switch (rhs.storage()) {
        case MatrixStorage::FullMatrix:
            cblas_sgemm(
                layout,
                transa,
                transb,
                M, N, K,
                1.0F,
                lhs.raw_cast<const float*>(),
                lda,
                rhs.raw_cast<const float*>(),
                ldb,
                0.0F,
                result.raw_cast<float*>(),
                ldc);
            break;


        default:
            throw std::runtime_error("matrx-matrix multiplications of these formats is not currently supported");
    }

}

static void triangular_matrix_matrix(ScalarMatrix &result, const ScalarMatrix &lhs, blas::BlasUpLo layout, const ScalarMatrix &rhs){}
static void banded_matrix_matrix(ScalarMatrix &result, const ScalarMatrix &lhs, const deg_t band, const ScalarMatrix &rhs){}

ScalarMatrix FloatBlas::matrix_matrix(const ScalarMatrix &lhs, const ScalarMatrix &rhs) {
    assert(lhs.type() == rhs.type() && lhs.type() == type());

    if (lhs.ncols() != rhs.nrows()) {
        throw std::invalid_argument("inner matrix dimensions must agree");
    };

    ScalarMatrix result(type(), lhs.nrows(), rhs.ncols());

    switch (lhs.storage()) {
        case MatrixStorage::FullMatrix:
            full_matrix_matrix(result, lhs, rhs);
            break;
        case MatrixStorage::UpperTriangular:
        case MatrixStorage::LowerTriangular:
            triangular_matrix_matrix(result, lhs, blas::to_blas_uplo(lhs.storage()), rhs);
            break;
        case MatrixStorage::Diagonal:
            banded_matrix_matrix(result, lhs, 1, rhs);
            break;
    }

    return result;
}
ScalarMatrix FloatBlas::solve_linear_system(const ScalarMatrix &coeff_matrix, const ScalarMatrix &target_matrix) {
    return BlasInterface::solve_linear_system(coeff_matrix, target_matrix);
}
ScalarArray FloatBlas::lls_qr(const ScalarMatrix &matrix, const ScalarArray &target) {
    return BlasInterface::lls_qr(matrix, target);
}
ScalarArray FloatBlas::lls_orth(const ScalarMatrix &matrix, const ScalarArray &target) {
    return BlasInterface::lls_orth(matrix, target);
}
ScalarArray FloatBlas::lls_svd(const ScalarMatrix &matrix, const ScalarArray &target) {
    return BlasInterface::lls_svd(matrix, target);
}
ScalarArray FloatBlas::lls_dcsvd(const ScalarMatrix &matrix, const ScalarArray &target) {
    return BlasInterface::lls_dcsvd(matrix, target);
}
ScalarArray FloatBlas::lse_grq(const ScalarMatrix &A, const ScalarMatrix &B, const ScalarArray &c, const ScalarArray &d) {
    return BlasInterface::lse_grq(A, B, c, d);
}
ScalarMatrix FloatBlas::glm_GQR(const ScalarMatrix &A, const ScalarMatrix &B, const ScalarArray &d) {
    return BlasInterface::glm_GQR(A, B, d);
}
EigenDecomposition FloatBlas::eigen_decomposition(const ScalarMatrix &matrix) {
    return BlasInterface::eigen_decomposition(matrix);
}
SingularValueDecomposition FloatBlas::svd(const ScalarMatrix &matrix) {
    return BlasInterface::svd(matrix);
}
