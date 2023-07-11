//
// Created by sam on 21/04/23.
//

#ifndef ROUGHPY_SCALAR_BLAS_IMPL_H
#define ROUGHPY_SCALAR_BLAS_IMPL_H

#include "owned_scalar_array.h"
#include "scalar.h"
#include "scalar_array.h"
#include "scalar_blas.h"
#include "scalar_matrix.h"
#include "scalar_type.h"

#include "scalar_blas_defs.h"

namespace rpy {
namespace scalars {

template <typename S>
class ScalarBlasImpl : public BlasInterface
{
    using MatrixCM
            = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    using MatrixRM
            = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MapTypeCM = Eigen::Map<MatrixCM>;
    using ConstMapTypeCM = Eigen::Map<const MatrixCM>;

    static MatrixCM to_matrix(const ScalarMatrix& arg)
    {
        MatrixCM result(arg.nrows(), arg.ncols());
        auto* optr = result.data();

        const auto* type = arg.type();

        if (arg.layout() != MatrixLayout::FStype) {
            ScalarMatrix tmp(arg.nrows(), arg.ncols(),
                             ScalarArray(type, optr, arg.size()),
                             MatrixStorage::FullMatrix, MatrixLayout::FStype);
            arg.to_full(tmp);
        } else {
            type->convert_copy(optr, arg, arg.size());
        }
        return result;
    }

    ScalarMatrix from_matrix(const MatrixCM& arg) const
    {
        ScalarMatrix result(type(), arg.rows(), arg.cols(),
                            MatrixStorage::FullMatrix, MatrixLayout::FStype);
        type()->convert_copy(result, {type(), arg.data()}, arg.size());
        return result;
    }

public:
    void transpose(ScalarMatrix& matrix) const override;
    OwnedScalarArray vector_axpy(const ScalarArray& x, const Scalar& a,
                                 const ScalarArray& y) override;
    Scalar dot_product(const ScalarArray& lhs, const ScalarArray& rhs) override;
    Scalar L1Norm(const ScalarArray& vector) override;
    Scalar L2Norm(const ScalarArray& vector) override;
    Scalar LInfNorm(const ScalarArray& vector) override;
    OwnedScalarArray matrix_vector(const ScalarMatrix& matrix,
                                   const ScalarArray& vector) override;
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

template <typename S>
void ScalarBlasImpl<S>::transpose(ScalarMatrix& matrix) const
{}
template <typename S>
OwnedScalarArray ScalarBlasImpl<S>::vector_axpy(const ScalarArray& x,
                                                const Scalar& a,
                                                const ScalarArray& y)
{

    return {};
}
template <typename S>
Scalar ScalarBlasImpl<S>::dot_product(const ScalarArray& lhs,
                                      const ScalarArray& rhs)
{
    return BlasInterface::dot_product(lhs, rhs);
}
template <typename S>
Scalar ScalarBlasImpl<S>::L1Norm(const ScalarArray& vector)
{
    return BlasInterface::L1Norm(vector);
}
template <typename S>
Scalar ScalarBlasImpl<S>::L2Norm(const ScalarArray& vector)
{
    return BlasInterface::L2Norm(vector);
}
template <typename S>
Scalar ScalarBlasImpl<S>::LInfNorm(const ScalarArray& vector)
{
    return BlasInterface::LInfNorm(vector);
}
template <typename S>
OwnedScalarArray ScalarBlasImpl<S>::matrix_vector(const ScalarMatrix& matrix,
                                                  const ScalarArray& vector)
{
    return BlasInterface::matrix_vector(matrix, vector);
}
template <typename S>
ScalarMatrix ScalarBlasImpl<S>::matrix_matrix(const ScalarMatrix& lhs,
                                              const ScalarMatrix& rhs)
{
    return from_matrix(to_matrix(lhs) * to_matrix(rhs));
}
template <typename S>
ScalarMatrix
ScalarBlasImpl<S>::solve_linear_system(const ScalarMatrix& coeff_matrix,
                                       const ScalarMatrix& target_matrix)
{
    auto coeff = to_matrix(coeff_matrix);
    auto result = coeff.householderQr().solve(to_matrix(target_matrix));
    return from_matrix(result);
}
template <typename S>
OwnedScalarArray ScalarBlasImpl<S>::lls_qr(const ScalarMatrix& matrix,
                                           const ScalarArray& target)
{
    return BlasInterface::lls_qr(matrix, target);
}
template <typename S>
OwnedScalarArray ScalarBlasImpl<S>::lls_orth(const ScalarMatrix& matrix,
                                             const ScalarArray& target)
{
    return BlasInterface::lls_orth(matrix, target);
}
template <typename S>
OwnedScalarArray ScalarBlasImpl<S>::lls_svd(const ScalarMatrix& matrix,
                                            const ScalarArray& target)
{
    return BlasInterface::lls_svd(matrix, target);
}
template <typename S>
OwnedScalarArray ScalarBlasImpl<S>::lls_dcsvd(const ScalarMatrix& matrix,
                                              const ScalarArray& target)
{
    return BlasInterface::lls_dcsvd(matrix, target);
}
template <typename S>
OwnedScalarArray
ScalarBlasImpl<S>::lse_grq(const ScalarMatrix& A, const ScalarMatrix& B,
                           const ScalarArray& c, const ScalarArray& d)
{
    return BlasInterface::lse_grq(A, B, c, d);
}
template <typename S>
ScalarMatrix ScalarBlasImpl<S>::glm_GQR(const ScalarMatrix& A,
                                        const ScalarMatrix& B,
                                        const ScalarArray& d)
{
    return BlasInterface::glm_GQR(A, B, d);
}
template <typename S>
EigenDecomposition
ScalarBlasImpl<S>::eigen_decomposition(const ScalarMatrix& matrix)
{
    return BlasInterface::eigen_decomposition(matrix);
}
template <typename S>
SingularValueDecomposition ScalarBlasImpl<S>::svd(const ScalarMatrix& matrix)
{
    return BlasInterface::svd(matrix);
}

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALAR_BLAS_IMPL_H
