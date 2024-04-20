//
// Created by sam on 1/29/24.
//

#ifndef ROUGHPY_ALGEBRA_ALGEBRA_H
#define ROUGHPY_ALGEBRA_ALGEBRA_H

#include "graded_vector.h"
#include "roughpy_algebra_export.h"
#include "vector.h"

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

namespace rpy {
namespace algebra {

namespace dtl {

template <typename Multiplication>
struct MultiplicationTraits {

    static bool basis_compatibility_check(const Basis& basis)
    {
        return Multiplication::basis_compatibility_checK(basis);
    }

    static void fma(Vector& out, const Vector& left, const Vector& right)
    {
        Multiplication::fma(out, left, right);
    }
    static void multiply_into(Vector& out, const Vector& other)
    {
        Multiplication::multiply_into(out, other);
    }
};

}// namespace dtl

/**
 * @brief The Algebra class represents an algebra in a mathematical context.
 *
 * This class extends the Base class and provides functionality for performing
 * algebraic operations.
 */
template <typename Multiplication, typename Base = Vector>
class Algebra : public Base
{

protected:
    using multiplication_traits = dtl::MultiplicationTraits<Multiplication>;

    static bool basis_compatibility_check(const Basis& basis) noexcept
    {
        return multiplication_traits::basis_compatibility_check(basis)
                && Base::basis_compatibility_check(basis);
    }

public:
    using multiplication_type = Multiplication;

    using Base::Base;

    /**
     * @brief Multiply the algebra by another vector on the right.
     *
     * Multiply *this on the right by other result = (*this) * other;
     *
     * @param other The vector to multiply with.
     * @return Algebra The result of the multiplication.
     */
    RPY_NO_DISCARD Algebra right_multiply(const Vector& other) const;

    /**
     * @brief Multiply the algebra by another vector on left.
     *
     * Multiply *this on the left by other result = other * (*this).
     *
     * @param other The vector to multiply with.
     * @return Algebra The result of the multiplication.
     */
    RPY_NO_DISCARD Algebra left_multiply(const Vector& other) const;

    /**
     * @brief Multiply the algebra in place by another vector.
     *
     * This method multiplies the current algebra object by another vector
     * and stores the result back in the algebra object itself.
     *
     * @param other The vector to multiply with.
     * @return The reference to the modified algebra object after
     * multiplication.
     *
     * @see Algebra<Multiplication, Base>
     * @see multiplication_traits::multiply_into()
     */
    Algebra& multiply_inplace(const Vector& other);
};

template <typename Multiplication>
using GradedAlgebra = Algebra<Multiplication, GradedVector<>>;

template <typename Multiplication, typename Base>
Algebra<Multiplication, Base>
Algebra<Multiplication, Base>::right_multiply(const Vector& other) const
{
    Algebra result(this->basis(), this->scalar_type());
    multiplication_traits::fma(result, *this, other);
    return result;
}

template <typename Multiplication, typename Base>
Algebra<Multiplication, Base>
Algebra<Multiplication, Base>::left_multiply(const Vector& other) const
{
    Algebra result(this->basis(), this->scalar_type());
    multiplication_traits::fma(result, other, *this);
    return result;
}

template <typename Multiplication, typename Base>
Algebra<Multiplication, Base>&
Algebra<Multiplication, Base>::multiply_inplace(const Vector& other)
{
    multiplication_traits::multiply_into(*this, other);
    return *this;
}

namespace dtl {

template <typename, typename = void>
constexpr bool is_algebra = false;

template <typename T>
constexpr bool is_algebra<
        T,
        void_t<typename T::multiplication_type,
               decltype(std::declval<const T&>()
                                .left_multiply(std::declval<const Vector&>())),
               decltype(std::declval<const T&>()
                                .right_multiply(std::declval<const Vector&>())),
               decltype(std::declval<T&>().multiply_inplace(std::declval<
                                                            const Vector&>()))>>
        = true;

}// namespace dtl

template <typename A>
RPY_NO_DISCARD enable_if_t<dtl::is_algebra<A>, A>
operator*(const A& left, const Vector& right)
{
    return A(left.right_multiply(right));
}

template <typename A>
RPY_NO_DISCARD enable_if_t<dtl::is_algebra<A>, A>
operator*(const Vector& left, const A& right)
{
    return A(right.left_multiply(left));
}

template <typename A>
RPY_NO_DISCARD enable_if_t<dtl::is_algebra<A>, A&>
operator*=(A& left, const Vector& right)
{
    left.multiply_inplace(right);
    return left;
}

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_ALGEBRA_H
