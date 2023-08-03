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

#ifndef ROUGHPY_SCALARS_BLAS_H_
#define ROUGHPY_SCALARS_BLAS_H_

#include "scalar_matrix.h"
#include "scalars_fwd.h"

#include <memory>
#include <mutex>

namespace rpy {
namespace scalars {

struct QRFactorization {
    ScalarMatrix Q;
    ScalarMatrix R;
};

struct EigenDecomposition {
    ScalarMatrix Lambda;
    ScalarMatrix U;
};

struct SingularValueDecomposition {
    ScalarMatrix U;
    ScalarMatrix Sigma;
    ScalarMatrix VHermitian;
};

class RPY_EXPORT BlasInterface
{
    const ScalarType* p_type;

    /// Must be acquired by all computation routines.
    std::recursive_mutex m_lock;

protected:
    inline void type_check(const ScalarMatrix& matrix)
    {
        RPY_CHECK(matrix.type() == p_type);
    }

    inline void matrix_product_check(matrix_dim_t A_ncols, matrix_dim_t B_nrows)
    {
        RPY_CHECK(A_ncols == B_nrows);
    }

    RPY_NO_DISCARD inline std::lock_guard<std::recursive_mutex> lock()
    {
        return std::lock_guard<std::recursive_mutex>(m_lock);
    }

public:
    RPY_NO_DISCARD const ScalarType* type() const noexcept { return p_type; }

    explicit BlasInterface(const ScalarType* type);

    virtual ~BlasInterface();

    RPY_NO_DISCARD std::unique_ptr<BlasInterface> clone() const;

    virtual void transpose(ScalarMatrix& matrix) const;

    // BLAS
    // Level 1
    RPY_NO_DISCARD virtual OwnedScalarArray
    vector_axpy(const ScalarArray& x, const Scalar& a, const ScalarArray& y);
    RPY_NO_DISCARD virtual Scalar
    dot_product(const ScalarArray& lhs, const ScalarArray& rhs);
    RPY_NO_DISCARD virtual Scalar L1Norm(const ScalarArray& vector);
    RPY_NO_DISCARD virtual Scalar L2Norm(const ScalarArray& vector);
    RPY_NO_DISCARD virtual Scalar LInfNorm(const ScalarArray& vector);

    // Level 2
    /**
     * @brief Compute y := alpha*A*x + beta*y
     * @param y
     * @param A
     * @param x
     * @param alpha
     * @param beta
     */
    RPY_NO_DISCARD virtual void
    gemv(ScalarMatrix& y, const ScalarMatrix& A, const ScalarMatrix& x,
         const Scalar& alpha, const Scalar& beta);
    // Level 3

    /**
     * @brief Compute C := alpha*A*B + beta*C
     * @param C
     * @param A
     * @param B
     * @param alpha
     * @param beta
     */
    RPY_NO_DISCARD virtual void
    gemm(ScalarMatrix& C, const ScalarMatrix& A, const ScalarMatrix& B,
         const Scalar& alpha, const Scalar& beta);

    // LAPACK

    /**
     * @brief Solver the linear System A*X=B, result written into X
     * @param A Matrix of coefficients
     * @param B Target vector(s)
     */
    RPY_NO_DISCARD virtual void gesv(ScalarMatrix& A, ScalarMatrix& B);

    //    RPY_NO_DISCARD virtual QRFactorization geqrf(const ScalarMatrix&
    //    matrix); RPY_NO_DISCARD virtual QRFactorization geqpf(const
    //    ScalarMatrix& matrix);

    // Eigenvalues
    /**
     * @brief Compute the eigenvalues and (optionally) eigenvectors of a
     * symmetric matrix.
     * @param matrix
     * @param eigenvectors
     * @return
     */
    RPY_NO_DISCARD virtual EigenDecomposition
    syev(ScalarMatrix& A, bool eigenvectors);

    /**
     * @brief Compute the eigenvalues and (optionally) eigenvectors of a
     * general matrix.
     * @param matrix
     * @param eigenvectors
     * @return
     */
    RPY_NO_DISCARD virtual EigenDecomposition
    geev(ScalarMatrix& A, bool eigenvectors);

    // Singular value decomposition

    /**
     * @brief Compute the singular value decomposition of A = U*Sigma*VT
     * @param A
     * @param return_U
     * @param return_VT
     * @return
     */
    RPY_NO_DISCARD virtual SingularValueDecomposition
    gesvd(ScalarMatrix& A, bool return_U, bool return_VT);

    /**
     * @brief Compute the singular value decomposition of A = U*Sigma*VT
     * using a divide-and-conquer algorithm.
     * @param matrix
     * @param return_U
     * @param return_VT
     * @return
     */
    RPY_NO_DISCARD virtual SingularValueDecomposition
    gesdd(ScalarMatrix& A, bool return_U, bool return_VT);

    // Least Squares
    /**
     * @brief Compute the least squares solution to A*x=B by minimizing
     * ||A*x-B||^2. Results are written into x
     * @param A
     * @param x
     * @return
     */
    RPY_NO_DISCARD virtual void gels(ScalarMatrix& A, ScalarMatrix& b);

    /**
     * @brief Compute the least squares solution to A*x=B by minimizing
     * ||A*x-B||^2 using complete orthogonal factorization of A. Results are
     * written into x
     * @param A
     * @param x
     * @return
     */
    RPY_NO_DISCARD virtual ScalarMatrix gelsy(ScalarMatrix& A, ScalarMatrix& b);

    /**
     * @brief Compute the least squares solution to A*x=B by minimizing
     * ||A*x-B||^2 using the singular value decomposition. Results are
     * written into x
     * @param A
     * @param x
     * @return
     */
    RPY_NO_DISCARD virtual ScalarMatrix gelss(ScalarMatrix& A, ScalarMatrix& b);

    /**
     * @brief Compute the least squares solution to A*x=B by minimizing
     * ||A*x-B||^2 using a divide-and-conquer singular value decomposition
     * method. Results are written into x
     * @param A
     * @param x
     * @return
     */
    RPY_NO_DISCARD virtual ScalarMatrix gelsd(ScalarMatrix& A, ScalarMatrix& b);
};

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_BLAS_H_
