//
// Created by sam on 16/02/24.
//

#ifndef ROUGHPY_ALGEBRA_FREE_TENSOR_H
#define ROUGHPY_ALGEBRA_FREE_TENSOR_H

#include "algebra.h"

namespace rpy {
namespace algebra {


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
class FreeTensor : public AlgebraBase<FreeTensor>
{

public:
    static FreeTensor new_like(const FreeTensor& arg) noexcept;
    static FreeTensor clone(const FreeTensor& arg) noexcept;
    static FreeTensor from(Vector&& arg) noexcept;
    static FreeTensor unit_like(const FreeTensor& arg) noexcept;

    FreeTensor();

    FreeTensor(const FreeTensor& arg) = default;
    FreeTensor(FreeTensor&& arg) noexcept = default;

    FreeTensor& operator=(const FreeTensor&) = default;
    FreeTensor& operator=(FreeTensor&&) noexcept = default;

    FreeTensor&
    fma(const Vector& lhs,
        const Vector& rhs,
        const ops::Operator& op// NOLINT(*-identifier-length)
    );
    FreeTensor&
    fma(const Vector& lhs,
        const Vector& rhs,
        const ops::Operator& op,// NOLINT(*-identifier-length)
        deg_t max_degree,
        deg_t lhs_min_deg = 0,
        deg_t rhs_min_deg = 0);

    FreeTensor& multiply_inplace(
            const Vector& rhs,
            const ops::Operator& op// NOLINT(*-identifier-length)
    );
    FreeTensor& multiply_inplace(
            const Vector& rhs,
            const ops::Operator& op,// NOLINT(*-identifier-length)
            deg_t max_degree,
            deg_t lhs_min_deg = 0,
            deg_t rhs_min_deg = 0
    );

    /**
     * @brief Calculates the exponential value of a FreeTensor object.
     *
     * This method calculates the exponential value of the current FreeTensor
     * object and returns the result as a new FreeTensor object.
     *
     * @return A new FreeTensor object representing the exponential value.
     */
    RPY_NO_DISCARD FreeTensor exp() const;

    /**
     * @brief Calculates the logarithm of a FreeTensor object.
     *
     * This method computes the logarithm of a FreeTensor object and returns the
     * result as a new FreeTensor object.
     *
     * @return The logarithm of the FreeTensor object.
     */
    RPY_NO_DISCARD FreeTensor log() const;

    /**
     * @brief Calculates the antipode of a FreeTensor object.
     *
     * This method computes the antipode of a FreeTensor object and returns
     * the result as a new FreeTensor object.
     *
     * @return The antipode of the FreeTensor object.
     */
    RPY_NO_DISCARD FreeTensor antipode() const;

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
    RPY_NO_DISCARD FreeTensor fused_multiply_exp(const FreeTensor& other) const;

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

template <typename T, typename SFINAE = void>
inline constexpr bool is_tensor = false;

template <typename T>
inline constexpr bool is_tensor<
        T,
        enable_if_t<is_same_v<FreeTensor, T> || is_base_of_v<FreeTensor, T>>>
        = true;

template <typename T>
enable_if_t<is_tensor<T>, T> antipode(const T& arg)
{
    return arg.antipode();
}

template <typename T>
enable_if_t<is_tensor<T>, T> exp(const T& arg)
{
    return arg.exp();
}

template <typename T>
enable_if_t<is_tensor<T>, T> log(const T& arg)
{
    return arg.log();
}

template <typename T>
enable_if_t<is_tensor<T>, T> fmexp(const T& multiply, const T& exponent)
{
    return multiply.fused_multiply_exp(exponent);
}

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_FREE_TENSOR_H
