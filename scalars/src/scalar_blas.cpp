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
// Created by user on 18/04/23.
//

#include <roughpy/scalars/scalar_blas.h>

#include <roughpy/scalars/owned_scalar_array.h>
#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_array.h>
#include <roughpy/scalars/scalar_type.h>

using namespace rpy;
using namespace rpy::scalars;

BlasInterface::BlasInterface(const ScalarType* type) : p_type(type) {}
BlasInterface::~BlasInterface() = default;
std::unique_ptr<BlasInterface> BlasInterface::clone() const
{
    return std::unique_ptr<BlasInterface>();
}
void BlasInterface::transpose(ScalarMatrix& matrix) const {}
OwnedScalarArray BlasInterface::vector_axpy(const ScalarArray& x,
                                            const Scalar& a,
                                            const ScalarArray& y)
{
    return OwnedScalarArray();
}
Scalar BlasInterface::dot_product(const ScalarArray& lhs,
                                  const ScalarArray& rhs)
{
    return Scalar();
}
Scalar BlasInterface::L1Norm(const ScalarArray& vector) { return Scalar(); }
Scalar BlasInterface::L2Norm(const ScalarArray& vector) { return Scalar(); }
Scalar BlasInterface::LInfNorm(const ScalarArray& vector) { return Scalar(); }
OwnedScalarArray BlasInterface::matrix_vector(const ScalarMatrix& matrix,
                                              const ScalarArray& vector)
{
    return OwnedScalarArray();
}
ScalarMatrix BlasInterface::matrix_matrix(const ScalarMatrix& lhs,
                                          const ScalarMatrix& rhs)
{
    return ScalarMatrix(nullptr, 0, 0);
}
ScalarMatrix
BlasInterface::solve_linear_system(const ScalarMatrix& coeff_matrix,
                                   const ScalarMatrix& target_matrix)
{
    return ScalarMatrix(nullptr, 0, 0);
}
OwnedScalarArray BlasInterface::lls_qr(const ScalarMatrix& matrix,
                                       const ScalarArray& target)
{
    return OwnedScalarArray();
}
OwnedScalarArray BlasInterface::lls_orth(const ScalarMatrix& matrix,
                                         const ScalarArray& target)
{
    return OwnedScalarArray();
}
OwnedScalarArray BlasInterface::lls_svd(const ScalarMatrix& matrix,
                                        const ScalarArray& target)
{
    return OwnedScalarArray();
}
OwnedScalarArray BlasInterface::lls_dcsvd(const ScalarMatrix& matrix,
                                          const ScalarArray& target)
{
    return OwnedScalarArray();
}
OwnedScalarArray BlasInterface::lse_grq(const ScalarMatrix& A,
                                        const ScalarMatrix& B,
                                        const ScalarArray& c,
                                        const ScalarArray& d)
{
    return OwnedScalarArray();
}
ScalarMatrix BlasInterface::glm_GQR(const ScalarMatrix& A,
                                    const ScalarMatrix& B, const ScalarArray& d)
{
    return ScalarMatrix(nullptr, 0, 0);
}
EigenDecomposition
BlasInterface::eigen_decomposition(const ScalarMatrix& matrix)
{
    return {ScalarMatrix(p_type, 0, 0), ScalarMatrix(p_type, 0, 0)};
}
SingularValueDecomposition BlasInterface::svd(const ScalarMatrix& matrix)
{
    return {
            ScalarMatrix(p_type, 0, 0),
            ScalarMatrix(p_type, 0, 0),
            ScalarMatrix(p_type, 0, 0),
    };
}
