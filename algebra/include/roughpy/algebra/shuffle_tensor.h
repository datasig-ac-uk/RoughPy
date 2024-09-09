//
// Created by sam on 16/02/24.
//

#ifndef ROUGHPY_ALGEBRA_SHUFFLE_TENSOR_H
#define ROUGHPY_ALGEBRA_SHUFFLE_TENSOR_H

#include "algebra.h"
#include "roughpy_algebra_export.h"

namespace rpy {
namespace algebra {

class ShuffleTensorMultiplication : public RcBase<ShuffleTensorMultiplication>
{
public:
    static bool basis_compatibility_check(const Basis& basis) noexcept;
    static void fma(Vector& out, const Vector& left, const Vector& right);
    static void multiply_into(Vector& out, const Vector& right);
};

/**
 * @class ShuffleTensor
 * @brief A class representing a Shuffle Tensor algebra.
 *
 * ShuffleTensor is a concrete implementation of the UnitalAlgebra class.
 * It inherits from UnitalAlgebra and defines the ShuffleTensorMultiplication
 * as its multiplication operation and GradedAlgebra as its grading structure.
 *
 * ShuffleTensor algebra is used in mathematics for studying combinatorics,
 * homotopy theory, and algebraic topology. It is a graded algebra that captures
 * the property of shuffling of tensors.
 *
 * The ShuffleTensor class provides various functions and operations for
 * manipulating ShuffleTensor objects, such as tensor product, multiplication,
 * and grading structure.
 */
class ROUGHPY_ALGEBRA_EXPORT ShuffleTensor : public AlgebraBase<ShuffleTensor>
{
public:
    static ShuffleTensor new_like(const ShuffleTensor& arg) noexcept;
    static ShuffleTensor clone(const ShuffleTensor& arg) noexcept;
    static ShuffleTensor
    from_like(const ShuffleTensor& like, Vector&& data) noexcept;

    ShuffleTensor();

    ShuffleTensor&
    fma(const Vector& lhs, const Vector& rhs, const ops::Operator& op);
    ShuffleTensor& inplace_multiply(const Vector& rhs, const ops::Operator& op);
};

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_SHUFFLE_TENSOR_H
