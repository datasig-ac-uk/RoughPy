//
// Created by sam on 1/29/24.
//

#ifndef ROUGHPY_ALGEBRA_ALGEBRA_H
#define ROUGHPY_ALGEBRA_ALGEBRA_H

#include "graded_vector.h"
#include "roughpy_algebra_export.h"
#include "vector.h"

#include <roughpy/core/errors.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include <roughpy/device_support/operators.h>

namespace rpy {
namespace algebra {

namespace dtl {

namespace ops = devices::operators;

template <typename M>
inline constexpr bool is_graded = M::is_graded;

template <typename M, typename Op, typename SFINAE = void>
struct HasFMA {
    void operator()(
            const M& mul,
            Vector& out,
            const Vector& left,
            const Vector& right,
            Op&& op
    ) const
    {
        if constexpr (is_graded<M>) {
            const auto basis = out.basis();
            const auto max_deg = basis->max_degree();
            for (const auto& lhs_item : left) {
                for (const auto& rhs_item : right) {
                    if (basis->degree(lhs_item->first)
                                + basis->degree(rhs_item->first)
                        <= max_deg) {
                        out.add_scal_mul(
                                mul.key_product(
                                        lhs_item->first,
                                        rhs_item->first
                                ),
                                op(lhs_item->second, rhs_item->second)
                        );
                    }
                }
            }
        } else {
            for (const auto& lhs_item : left) {
                for (const auto& rhs_item : right) {
                    out.add_scal_mul(
                            mul.key_product(lhs_item->first, rhs_item->first),
                            op(lhs_item->second, rhs_item->second)
                    );
                }
            }
        }
    }
};

template <typename M, typename Op>
struct HasFMA<
        M,
        Op,
        void_t<decltype(std::declval<const M&>()
                                .fma(std::declval<Vector&>(),
                                     std::declval<const Vector&>(),
                                     std::declval<const Vector&>(),
                                     std::declval<Op&&>()))>> {
    void operator()(
            const M& mul,
            Vector& out,
            const Vector& left,
            const Vector& right,
            Op&& op
    ) const
    {
        mul.fma(out, left, right, std::forward<Op>(op));
    }
};

template <typename M, typename Op, typename SFINAE = void>
struct HasDenseFMA : HasFMA<M, Op> {
};

template <typename M, typename Op>
struct HasDenseFMA<
        M,
        Op,
        void_t<decltype(std::declval<const M&>().fma_dense(
                std::declval<Vector&>(),
                std::declval<const Vector&>(),
                std::declval<const Vector&>(),
                std::declval<Op&&>()
        ))>> : HasFMA<M, Op> {
    void operator()(
            const M& mul,
            Vector& out,
            const Vector& left,
            const Vector& right,
            Op&& op
    ) const
    {
        if (out.is_dense() && left.is_dense() && right.is_dense()) {
            mul.fma_dense(out, left, right, std::forward<Op>(op));
        } else {
            HasFMA<M,
                   Op>::operator()(mul, out, left, right, std::forward<Op>(op));
        }
    }
};

template <typename M, typename Op, typename SFINAE = void>
struct HasInplace {
    void
    operator()(const M& mul, Vector& left, const Vector& right, Op&& op) const
    {
        HasDenseFMA<M, Op> fma;
        Vector tmp(left);
        fma(mul, tmp, left, right, std::forward<Op>(op));
        std::swap(left, tmp);
    }
};

template <typename M, typename Op>
struct HasInplace<
        M,
        Op,
        void_t<decltype(std::declval<const M&>().multiply_inplace(
                std::declval<Vector&>(),
                std::declval<const Vector&>(),
                std::declval<Op&&>()
        ))>> {
    void
    operator()(const M& mul, Vector& left, const Vector& right, Op&& op) const
    {
        mul.multiply_inplace(left, right, std::forward<Op>(op));
    }
};

template <typename M, typename Op, typename SFINAE = void>
struct HasDenseInplace : HasInplace<M, Op> {
};

template <typename M, typename Op>
struct HasDenseInplace<
        M,
        Op,
        void_t<decltype(std::declval<const M&>().multiply_inplace_dense(
                std::declval<Vector&>(),
                std::declval<const Vector&>(),
                std::declval<Op&&>()
        ))>> : HasInplace<M, Op> {
    void
    operator()(const M& mul, Vector& left, const Vector& right, Op&& op) const
    {
        if (left.is_dense() && right.is_dense()) {
            mul.multiply_inplace_dense(left, right, std::forward<Op>(op));
        } else {
            HasInplace<M, Op>::operator()(
                    mul,
                    left,
                    right,
                    std::forward<Op>(op)
            );
        }
    }
};

template <typename Multiplication>
/**
 * @brief Class containing static methods for multiplication operations on an
 * algebra.
 *
 * The MultiplicationTraits class provides static methods for performing various
 * multiplication operations on an algebra. These operations include multiplying
 * two vectors, multiplying a vector by a scalar, and performing fused
 * multiply-add operations.
 *
 * Users can use the methods in this class directly or through the
 * multiplication_traits typedef.
 */
struct MultiplicationTraits {

    /**
     * @brief Check if the given basis is compatible with the multiplication
     * operation.
     *
     * This method checks if the given basis is compatible with the
     * multiplication operation. A basis is considered compatible if it
     * satisfies certain conditions defined by the multiplication operation.
     *
     * @param basis The basis to check for compatibility.
     * @return True if the basis is compatible with the multiplication
     * operation, false otherwise.
     */
    static bool basis_compatibility_check(const Basis& basis)
    {
        return Multiplication::basis_compatibility_check(basis);
    }

    static void
    fma(const Multiplication& mul,
        Vector& out,
        const Vector& left,
        const Vector& right)
    {
        using Op = ops::Identity<scalars::Scalar>;
        HasDenseFMA<Multiplication, Op> fma;
        fma(mul, out, left, right, Op());
    }

    static void
    fms(const Multiplication& mul,
        Vector& out,
        const Vector& left,
        const Vector& right)
    {
        using Op = ops::Uminus<scalars::Scalar>;
        HasDenseFMA<Multiplication, Op> fma;
        fma(mul, out, left, right, Op());
    }

    static void
    fma_pm(const Multiplication& mul,
           Vector& out,
           const Vector& left,
           const Vector& right,
           const scalars::Scalar& multiplier)
    {
        using Op = ops::RightScalarMultiply<scalars::Scalar>;
        HasDenseFMA<Multiplication, Op> fma;
        fma(mul, out, left, right, Op(multiplier));
    }

    static void
    fms_pm(const Multiplication& mul,
           Vector& out,
           const Vector& left,
           const Vector& right,
           const scalars::Scalar& multiplier)
    {
        using Op = ops::RightScalarMultiply<scalars::Scalar>;
        HasDenseFMA<Multiplication, Op> fma;
        fma(mul, out, left, right, Op(-multiplier));
    }

    static void
    fma_pd(const Multiplication& mul,
           Vector& out,
           const Vector& left,
           const Vector& right,
           const scalars::Scalar& divisor)
    {
        using Op = ops::RightScalarMultiply<scalars::Scalar>;
        HasDenseFMA<Multiplication, Op> fma;
        fma(mul, out, left, right, Op(devices::math::reciprocal(divisor)));
    }

    static void
    fms_pd(const Multiplication& mul,
           Vector& out,
           const Vector& left,
           const Vector& right,
           const scalars::Scalar& divisor)
    {
        using Op = ops::RightScalarMultiply<scalars::Scalar>;
        HasDenseFMA<Multiplication, Op> fma;
        fma(mul, out, left, right, Op(-devices::math::reciprocal(divisor)));
    }

    static void multiply_into(Vector& out, const Vector& other)
    {
        Multiplication::multiply_into(out, other);
    }

    static void multiply_into_pm(
            const Multiplication& mul,
            Vector& left,
            const Vector& right,
            const scalars::ScalarCRef scalar
    )
    {
        using Op = ops::RightScalarMultiply<scalars::Scalar>;
        HasInplace<Multiplication, Op> inplace;
        inplace(mul, left, right, Op(scalars::Scalar(scalar)));
    }

    static void multiply_into_pd(
            const Multiplication& mul,
            Vector& left,
            const Vector& right,
            const scalars::ScalarCRef divisor
    )
    {
        using Op = ops::RightScalarMultiply<scalars::Scalar>;
        HasInplace<Multiplication, Op> inplace;
        inplace(mul, left, right, Op(devices::math::reciprocal(divisor)));
    }
};

}// namespace dtl

/**
 * @brief A class representing an algebra with multiplication operations.
 *
 * The Algebra class is a concrete implementation of an algebra.
 * It inherits from the Base class, which provides common functionality,
 * and is associated with a Multiplication object that is responsible for
 * performing multiplication operations on the Algebra.
 *
 * @see Base
 * @see Multiplication
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

private:
    Rc<const Multiplication> p_multiplication;

protected:
    Rc<const Multiplication> get_multiplication() const noexcept
    {
        return p_multiplication;
    }

public:
    /**
     * @brief Returns a const reference to the Multiplication object associated
     * with the Algebra.
     *
     * This method returns a const reference to the Multiplication object
     * associated with the Algebra. The Multiplication object is responsible for
     * performing multiplication operations on the Algebra.
     *
     * @return const Reference to the Multiplication object associated with the
     * Algebra.
     *
     * @see Algebra::right_multiply()
     * @see Algebra::left_multiply()
     * @see Algebra::multiply_inplace()
     * @see Algebra::add_multiply()
     * @see Algebra::post_scalar_multiply()
     * @see Algebra::post_scalar_divide()
     */
    const Multiplication& multiplication() const noexcept
    {
        return *p_multiplication;
    }

    using Base::Base;

    /**
     * @brief Multiply the algebra by another vector on the right.
     *
     * Multiply *this on the right by other result = (*this) * other;
     *
     * @param other The vector to multiply with.
     * @return Algebra The result of the multiplication.
     */
    RPY_NO_DISCARD Algebra right_multiply(const Vector& other) const
    {
        Algebra result(this->basis(), this->p_multiplication);
        multiplication_traits::fma(multiplication(), result, *this, other);
        return result;
    }

    /**
     * @brief Multiply the algebra by another vector on left.
     *
     * Multiply *this on the left by other result = other * (*this).
     *
     * @param other The vector to multiply with.
     * @return Algebra The result of the multiplication.
     */
    RPY_NO_DISCARD Algebra left_multiply(const Vector& other) const
    {
        Algebra result(this->basis(), this->p_multiplication);
        multiplication_traits::fma(multiplication(), result, other, *this);
        return result;
    }

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
    Algebra& multiply_inplace(const Vector& other)
    {
        multiplication_traits::multiply_into(multiplication(), *this, other);
        return *this;
    }

    /**
     * @brief Multiply two vectors and add the result to the current algebra
     * object.
     *
     * This method multiplies two vectors (left and right) using the provided
     * multiplication traits and adds the result to the current algebra object.
     *
     * @param left The left vector to multiply.
     * @param right The right vector to multiply.
     * @return The reference to the modified algebra object after adding the
     * multiplication result.
     *
     * @see multiplication_traits::fma()
     */
    Algebra& add_multiply(const Vector& left, const Vector& right)
    {
        multiplication_traits::fma(multiplication(), *this, left, right);
        return *this;
    }
    /**
     * @brief Multiply the current algebra object by two vectors and store the
     * result in the algebra object itself.
     *
     * This method multiplies the current algebra object (*this) by two vectors
     * (left and right) using the provided multiplication traits and stores the
     * result back in the algebra object itself.
     *
     * @param left The left vector to multiply.
     * @param right The right vector to multiply.
     * @return The reference to the modified algebra object after the
     * multiplication.
     *
     * @see multiplication_traits::fms()
     */
    Algebra& sub_multiply(const Vector& left, const Vector& right)
    {
        multiplication_traits::fms(multiplication(), *this, left, right);
        return *this;
    }

    /**
     * @brief Multiply the result of a post-scalar multiplication by another
     * vector.
     *
     * Multiply the result of the scalar multiplication by another vector, i.e.
     * result = (scalar * this) * other, where this is the current vector and
     * scalar is a scalar value.
     *
     * @param multiplier The scalar value to multiply with the post-scalar
     * multiplication result.
     * @param rhs The vector to multiply with the post-scalar multiplication
     * result.
     * @return Algebra The result of the multiplication.
     */
    Algebra&
    multiply_post_smul(const Vector& rhs, const scalars::ScalarCRef multiplier)
    {
        multiplication_traits::multiply_into_pm(rhs, std::move(multiplier));
        return *this;
    }

    /**
     * @brief Multiply the result of a post-scalar division operation by a
     * scalar value.
     *
     * This method multiplies the result of dividing the elements of a matrix by
     * a scalar value by a scalar value. It multiplies each element of the
     * matrix by the scalar.
     *
     * @param rhs The vector to multiply into the current value
     * @param divisor The scalar value to multiply the matrix by.
     */
    Algebra& multiply_post_sdiv(const Vector& rhs, scalars::ScalarCRef divisor)
    {
        multiplication_traits::multiply_into_pd(rhs, std::move(divisor));
        return *this;
    }
};

template <typename Multiplication>
using GradedAlgebra = Algebra<Multiplication, GradedVector<>>;

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
