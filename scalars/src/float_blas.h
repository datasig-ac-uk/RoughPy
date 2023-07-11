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

#ifndef ROUGHPY_SCALARS_SRC_FLOAT_BLAS_H
#define ROUGHPY_SCALARS_SRC_FLOAT_BLAS_H

#include <roughpy/scalars/scalar_blas.h>

namespace rpy {
namespace scalars {

class FloatBlas : public BlasInterface
{
public:
    explicit FloatBlas(const ScalarType* ctype) : BlasInterface(ctype) {}
    void transpose(ScalarMatrix& matrix) const override;
    OwnedScalarArray vector_axpy(const ScalarArray& x, const Scalar& a,
                                 const ScalarArray& y) override;
    Scalar dot_product(const ScalarArray& lhs, const ScalarArray& rhs) override;
    Scalar L1Norm(const ScalarArray& vector) override;
    Scalar L2Norm(const ScalarArray& vector) override;
    Scalar LInfNorm(const ScalarArray& vector) override;
    OwnedScalarArray matrix_vector(const ScalarMatrix& matrix,
                                   const ScalarArray& vector) override;

private:
public:
    ScalarMatrix matrix_matrix(const ScalarMatrix& lhs,
                               const ScalarMatrix& rhs) override;
    ScalarMatrix
    solve_linear_system(const ScalarMatrix& coeff_matrix,
                        const ScalarMatrix& target_matrix) override;
    OwnedScalarArray lls_qr(const ScalarMatrix& matrix,
                            const ScalarArray& target) override;
    OwnedScalarArray lls_orth(const ScalarMatrix& matrix,
                              const ScalarArray& target) override;
    OwnedScalarArray lls_svd(const ScalarMatrix& matrix,
                             const ScalarArray& target) override;
    OwnedScalarArray lls_dcsvd(const ScalarMatrix& matrix,
                               const ScalarArray& target) override;
    OwnedScalarArray lse_grq(const ScalarMatrix& A, const ScalarMatrix& B,
                             const ScalarArray& c,
                             const ScalarArray& d) override;
    ScalarMatrix glm_GQR(const ScalarMatrix& A, const ScalarMatrix& B,
                         const ScalarArray& d) override;
    EigenDecomposition eigen_decomposition(const ScalarMatrix& matrix) override;
    SingularValueDecomposition svd(const ScalarMatrix& matrix) override;
};

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SRC_FLOAT_BLAS_H
