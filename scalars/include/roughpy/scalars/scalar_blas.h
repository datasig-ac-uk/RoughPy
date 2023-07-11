#ifndef ROUGHPY_SCALARS_BLAS_H_
#define ROUGHPY_SCALARS_BLAS_H_

#include "scalar_matrix.h"
#include "scalars_fwd.h"

#include <memory>

namespace rpy {
namespace scalars {

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

public:
    RPY_NO_DISCARD
    const ScalarType* type() const noexcept { return p_type; }

    explicit BlasInterface(const ScalarType* type);

    virtual ~BlasInterface();

    RPY_NO_DISCARD
    std::unique_ptr<BlasInterface> clone() const;

    virtual void transpose(ScalarMatrix& matrix) const;

    // BLAS
    // Level 1
    RPY_NO_DISCARD
    virtual OwnedScalarArray vector_axpy(const ScalarArray& x, const Scalar& a,
                                         const ScalarArray& y);
    RPY_NO_DISCARD
    virtual Scalar dot_product(const ScalarArray& lhs, const ScalarArray& rhs);
    RPY_NO_DISCARD
    virtual Scalar L1Norm(const ScalarArray& vector);
    RPY_NO_DISCARD
    virtual Scalar L2Norm(const ScalarArray& vector);
    RPY_NO_DISCARD
    virtual Scalar LInfNorm(const ScalarArray& vector);

    // Level 2
    RPY_NO_DISCARD
    virtual OwnedScalarArray matrix_vector(const ScalarMatrix& matrix,
                                           const ScalarArray& vector);

    // Level 3
    RPY_NO_DISCARD
    virtual ScalarMatrix matrix_matrix(const ScalarMatrix& lhs,
                                       const ScalarMatrix& rhs);

    // LAPACK
    // Linear equations
    RPY_NO_DISCARD
    virtual ScalarMatrix solve_linear_system(const ScalarMatrix& coeff_matrix,
                                             const ScalarMatrix& target_matrix);

    // Least squares
    RPY_NO_DISCARD
    virtual OwnedScalarArray lls_qr(const ScalarMatrix& matrix,
                                    const ScalarArray& target);
    RPY_NO_DISCARD
    virtual OwnedScalarArray lls_orth(const ScalarMatrix& matrix,
                                      const ScalarArray& target);
    RPY_NO_DISCARD
    virtual OwnedScalarArray lls_svd(const ScalarMatrix& matrix,
                                     const ScalarArray& target);
    RPY_NO_DISCARD
    virtual OwnedScalarArray lls_dcsvd(const ScalarMatrix& matrix,
                                       const ScalarArray& target);

    RPY_NO_DISCARD
    virtual OwnedScalarArray lse_grq(const ScalarMatrix& A,
                                     const ScalarMatrix& B,
                                     const ScalarArray& c,
                                     const ScalarArray& d);
    RPY_NO_DISCARD
    virtual ScalarMatrix glm_GQR(const ScalarMatrix& A, const ScalarMatrix& B,
                                 const ScalarArray& d);

    // Eigenvalues and singular values
    RPY_NO_DISCARD
    virtual EigenDecomposition eigen_decomposition(const ScalarMatrix& matrix);
    RPY_NO_DISCARD
    virtual SingularValueDecomposition svd(const ScalarMatrix& matrix);
};

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_BLAS_H_
