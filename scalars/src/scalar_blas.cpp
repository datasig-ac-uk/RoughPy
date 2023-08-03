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
// Created by user on 18/04/23.
//

#include <roughpy/scalars/scalar_blas.h>

#include <roughpy/scalars/owned_scalar_array.h>
#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_array.h>
#include <roughpy/scalars/scalar_type.h>

using namespace rpy;
using namespace rpy::scalars;

// TODO: The definitions here are just to stop crashes. Needs work!

BlasInterface::BlasInterface(const ScalarType* type) : p_type(type) {}
BlasInterface::~BlasInterface() = default;

std::unique_ptr<BlasInterface> BlasInterface::clone() const
{
    return std::unique_ptr<BlasInterface>();
}
void BlasInterface::transpose(ScalarMatrix& matrix) const {}
OwnedScalarArray BlasInterface::vector_axpy(
        const ScalarArray& x, const Scalar& a, const ScalarArray& y
)
{
    return OwnedScalarArray();
}
Scalar
BlasInterface::dot_product(const ScalarArray& lhs, const ScalarArray& rhs)
{
    return Scalar();
}
Scalar BlasInterface::L1Norm(const ScalarArray& vector) { return Scalar(); }
Scalar BlasInterface::L2Norm(const ScalarArray& vector) { return Scalar(); }
Scalar BlasInterface::LInfNorm(const ScalarArray& vector) { return Scalar(); }
void BlasInterface::gemv(
        ScalarMatrix& y, const ScalarMatrix& A, const ScalarMatrix& x,
        const Scalar& alpha, const Scalar& beta
)
{}
void BlasInterface::gemm(
        ScalarMatrix& C, const ScalarMatrix& A, const ScalarMatrix& B,
        const Scalar& alpha, const Scalar& beta
)
{}
void BlasInterface::gesv(ScalarMatrix& A, ScalarMatrix& B) {}
EigenDecomposition BlasInterface::syev(ScalarMatrix& A, bool eigenvectors)
{
    return EigenDecomposition();
}
EigenDecomposition BlasInterface::geev(ScalarMatrix& A, bool eigenvectors)
{
    return EigenDecomposition();
}
SingularValueDecomposition
BlasInterface::gesvd(ScalarMatrix& A, bool return_U, bool return_VT)
{
    return SingularValueDecomposition();
}
SingularValueDecomposition
BlasInterface::gesdd(ScalarMatrix& A, bool return_U, bool return_VT)
{
    return SingularValueDecomposition();
}
void BlasInterface::gels(ScalarMatrix& A, ScalarMatrix& b) {}
ScalarMatrix BlasInterface::gelsy(ScalarMatrix& A, ScalarMatrix& b)
{
    return ScalarMatrix();
}
ScalarMatrix BlasInterface::gelss(ScalarMatrix& A, ScalarMatrix& b)
{
    return ScalarMatrix();
}
ScalarMatrix BlasInterface::gelsd(ScalarMatrix& A, ScalarMatrix& b)
{
    return ScalarMatrix();
}
