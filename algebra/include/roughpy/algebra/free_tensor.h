//
// Created by sam on 16/02/24.
//

#ifndef ROUGHPY_ALGEBRA_FREE_TENSOR_H
#define ROUGHPY_ALGEBRA_FREE_TENSOR_H

#include "algebra.h"
#include "unital_algebra.h"

namespace rpy {
namespace algebra {

/**
 * @class FreeTensorMultiplication
 *
 * @brief A class representing the multiplication operations for FreeTensor
 * objects.
 *
 * This class provides static methods to perform multiplication operations on
 * FreeTensor objects.
 */
class ROUGHPY_ALGEBRA_EXPORT FreeTensorMultiplication
{

public:
    /**
     * @brief Check if the given basis is compatible with the specific algebraic
     * operation.
     *
     * This method checks if the provided basis is compatible with the specific
     * algebraic operation. The compatibility depends on the requirements of
     * each operation.
     *
     * @param basis The basis to check for compatibility.
     * @return True if the basis is compatible, false otherwise.
     */
    static bool basis_compatibility_check(const Basis& basis) noexcept;

    static void fma(Vector& out, const Vector& left, const Vector& right);
    static void multiply_into(Vector& out, const Vector& right);
};

/**
 * @class FreeTensor
 *
 * @brief A class representing a free tensor.
 *
 * This class represents a free tensor, which is derived from the
 * GradedUnitalAlgebra class with FreeTensorMultiplication as the template
 * argument. It provides various methods to perform calculations on FreeTensor
 * objects, including calculating the exponential value, logarithm, antipode,
 * and performing fused multiplication and exponentiation operations.
 */
class FreeTensor : public GradedUnitalAlgebra<FreeTensorMultiplication>
{
    using base_t = GradedUnitalAlgebra<FreeTensorMultiplication>;

public:
    using base_t::base_t;

    /**
     * @brief Calculates the exponential value of a FreeTensor object.
     *
     * This method calculates the exponential value of the current FreeTensor
     * object and returns the result as a new FreeTensor object.
     *
     * @return A new FreeTensor object representing the exponential value.
     */
    FreeTensor exp() const;

    /**
     * @brief Calculates the logarithm of a FreeTensor object.
     *
     * This method computes the logarithm of a FreeTensor object and returns the
     * result as a new FreeTensor object.
     *
     * @return The logarithm of the FreeTensor object.
     */
    FreeTensor log() const;

    /**
     * @brief Calculates the antipode of a FreeTensor object.
     *
     * This method computes the antipode of a FreeTensor object and returns
     * the result as a new FreeTensor object.
     *
     * @return The antipode of the FreeTensor object.
     */
    FreeTensor antipode() const;

    /**
     * @brief Function to perform a fused multiplication and exponentiation
     * operation on a FreeTensor object.
     *
     * This function takes another FreeTensor object as input and returns a new
     * FreeTensor object that represents the result of performing a fused
     * multiplication and exponentiation operation.
     *
     * @param other The FreeTensor object to multiply and exponentiate with.
     * @return A new FreeTensor object that represents the result of the fused
     * multiplication and exponentiation operation.
     */
    FreeTensor fused_multiply_exp(const FreeTensor& other) const;

    /**
     * @brief Function to perform a fused multiplication and exponentiation in
     * place operation on a FreeTensor object.
     *
     * This function takes another FreeTensor object as input and performs a
     * fused multiplication and exponentiation operation in place on the current
     * FreeTensor object. The input FreeTensor object is multiplied and
     * exponentiated with the current FreeTensor object. The result is stored in
     * the current FreeTensor object itself.
     *
     * @param other The FreeTensor object to multiply and exponentiate with.
     * @return A reference to the current FreeTensor object after the operation.
     */
    FreeTensor& fused_multiply_exp_inplace(const FreeTensor& other);
};

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_FREE_TENSOR_H
