#ifndef ROUGHPY_SCALARS_BLAS_H_
#define ROUGHPY_SCALARS_BLAS_H_

#include "scalars_fwd.h"
#include "scalar_matrix.h"

#include <memory>

namespace rpy { namespace scalars {

struct EigenDecomposition {
    ScalarMatrix Lambda;
    ScalarMatrix U;
};

struct SingularValueDecomposition {
    ScalarMatrix U;
    ScalarMatrix Sigma;
    ScalarMatrix VHermitian;
};

class ROUGHPY_SCALARS_EXPORT BlasInterface {
    const ScalarType* p_type;
public:

    [[nodiscard]]
    const ScalarType* type() const noexcept { return p_type; }

    explicit BlasInterface(const ScalarType* type);

    virtual ~BlasInterface();

    std::unique_ptr<BlasInterface> clone() const;


    virtual void transpose(ScalarMatrix& matrix) const;

    // BLAS
    // Level 1
    virtual OwnedScalarArray vector_axpy(const ScalarArray& x, const Scalar& a, const ScalarArray& y);
    virtual Scalar dot_product(const ScalarArray& lhs, const ScalarArray& rhs);
    virtual Scalar L1Norm(const ScalarArray& vector);
    virtual Scalar L2Norm(const ScalarArray& vector);
    virtual Scalar LInfNorm(const ScalarArray& vector);

    // Level 2
    virtual OwnedScalarArray matrix_vector(const ScalarMatrix& matrix, const ScalarArray& vector);

    // Level 3
    virtual ScalarMatrix matrix_matrix(const ScalarMatrix& lhs, const ScalarMatrix& rhs);

    // LAPACK
    // Linear equations
    virtual ScalarMatrix solve_linear_system(const ScalarMatrix& coeff_matrix, const ScalarMatrix& target_matrix);

    // Least squares
    virtual OwnedScalarArray lls_qr(const ScalarMatrix& matrix, const ScalarArray& target);
    virtual OwnedScalarArray lls_orth(const ScalarMatrix& matrix, const ScalarArray& target);
    virtual OwnedScalarArray lls_svd(const ScalarMatrix& matrix, const ScalarArray& target);
    virtual OwnedScalarArray lls_dcsvd(const ScalarMatrix& matrix, const ScalarArray& target);

    virtual OwnedScalarArray lse_grq(const ScalarMatrix& A, const ScalarMatrix& B, const ScalarArray& c, const ScalarArray& d);
    virtual ScalarMatrix glm_GQR(const ScalarMatrix& A, const ScalarMatrix& B, const ScalarArray& d);

    // Eigenvalues and singular values
    virtual EigenDecomposition eigen_decomposition(const ScalarMatrix& matrix);
    virtual SingularValueDecomposition svd(const ScalarMatrix& matrix);


};




}}




#endif // ROUGHPY_SCALARS_BLAS_H_
