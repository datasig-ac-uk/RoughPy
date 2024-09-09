//
// Created by sam on 1/29/24.
//

#ifndef ROUGHPY_ALGEBRA_ALGEBRA_H
#define ROUGHPY_ALGEBRA_ALGEBRA_H

#include "roughpy_algebra_export.h"
#include "vector.h"

#include <roughpy/core/errors.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include <roughpy/devices/kernel_operators.h>

namespace rpy {
namespace algebra {

namespace ops = devices::operators;

template <typename Derived>
class AlgebraBase;

namespace dtl {

template <typename Derived>
constexpr Derived& cast_algebra_to_derived(AlgebraBase<Derived>& arg)
{
    static_assert(
            is_base_of_v<AlgebraBase<Derived>, Derived>,
            "Derived must be derived from AlgebraBase<Derived>"
    );

    return static_cast<Derived&>(arg);
}

template <typename Derived>
constexpr const Derived& cast_algebra_to_derived(const AlgebraBase<Derived>& arg
)
{
    static_assert(
            is_base_of_v<AlgebraBase<Derived>, Derived>,
            "Derived must be derived from AlgebraBase<Derived>"
    );

    return static_cast<const Derived&>(arg);
}

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
template <typename Derived>
class AlgebraBase
{
    Rc<Vector> p_vector;

    friend Derived;

    explicit AlgebraBase(Vector&& data) : p_vector(new Vector(std::move(data)))
    {}

public:
    using Scalar = scalars::Scalar;
    using ScalarCRef = scalars::ScalarCRef;
    using const_iterator = Vector::const_iterator;

protected:
    RPY_NO_DISCARD
    Derived& instance() noexcept { return dtl::cast_algebra_to_derived(*this); }
    RPY_NO_DISCARD
    const Derived& instance() const noexcept
    {
        return dtl::cast_algebra_to_derived(*this);
    }


public:
    RPY_NO_DISCARD Vector& as_vector() noexcept { return *p_vector; }
    RPY_NO_DISCARD const Vector& as_vector() const noexcept
    {
        return *p_vector;
    }

    RPY_NO_DISCARD bool is_dense() const noexcept
    {
        return p_vector->is_dense();
    }
    RPY_NO_DISCARD bool is_sparse() const noexcept
    {
        return p_vector->is_sparse();
    }
    RPY_NO_DISCARD VectorType vector_type() const noexcept
    {
        return p_vector->vector_type();
    }

    RPY_NO_DISCARD const BasisPointer& basis() const noexcept
    {
        return p_vector->basis();
    }
    RPY_NO_DISCARD scalars::TypePtr scalar_type() const noexcept
    {
        return p_vector->scalar_type();
    }

    RPY_NO_DISCARD const_iterator base_begin() const noexcept
    {
        return p_vector->base_begin();
    }
    RPY_NO_DISCARD const_iterator base_end() const noexcept
    {
        return p_vector->base_end();
    }

    RPY_NO_DISCARD const_iterator fibre_begin() const noexcept
    {
        return p_vector->fibre_begin();
    }
    RPY_NO_DISCARD const_iterator fibre_end() const noexcept
    {
        return p_vector->fibre_end();
    }

    RPY_NO_DISCARD const_iterator begin() const noexcept
    {
        return p_vector->begin();
    }

    RPY_NO_DISCARD const_iterator end() const noexcept
    {
        return p_vector->end();
    }

    RPY_NO_DISCARD ScalarCRef base_get(BasisKeyCRef key) const
    {
        return p_vector->base_get(std::move(key));
    }
    RPY_NO_DISCARD ScalarCRef fibre_get(BasisKeyCRef key) const
    {
        return p_vector->fibre_get(std::move(key));
    }

    RPY_NO_DISCARD Scalar base_get_mut(BasisKeyCRef key)
    {
        return p_vector->base_get_mut(std::move(key));
    }
    RPY_NO_DISCARD Scalar fibre_get_mut(BasisKeyCRef key)
    {
        return p_vector->fibre_get_mut(std::move(key));
    }

    RPY_NO_DISCARD Derived minus() const { return p_vector->minus(); }
    RPY_NO_DISCARD Derived left_smul(ScalarCRef scalar) const
    {
        return p_vector->left_smul(std::move(scalar));
    }
    RPY_NO_DISCARD Derived right_smul(ScalarCRef scalar) const
    {
        return p_vector->right_smul(std::move(scalar));
    }
    RPY_NO_DISCARD Derived sdiv(ScalarCRef scalar) const
    {
        const auto recip = devices::math::reciprocal(scalar);
        return right_smul(recip);
    }

    RPY_NO_DISCARD Derived add(const AlgebraBase& other) const
    {
        return p_vector->add(*other.p_vector);
    }
    RPY_NO_DISCARD Derived sub(const AlgebraBase& other) const
    {
        return p_vector->sub(*other.p_vector);
    }

    Derived& left_smul_inplace(ScalarCRef other)
    {
        p_vector->left_smul_inplace(std::move(other));
        return instance();
    }
    Derived& right_smul_inplace(ScalarCRef other)
    {
        p_vector->right_smul_inplace(std::move(other));
        return instance();
    }
    Derived& sdiv_inplace(ScalarCRef other)
    {
        const auto recip = devices::math::reciprocal(other);
        return right_smul_inplace(recip);
    }

    Derived& add_inplace(const AlgebraBase& other)
    {
        p_vector->add_inplace(*other.p_vector);
        return instance();
    }
    Derived& sub_inplace(const AlgebraBase& other)
    {
        p_vector->sub_inplace(*other.p_vector);
        return instance();
    }

    Derived& add_scal_mul(const AlgebraBase& other, ScalarCRef scalar)
    {
        p_vector->add_scal_mul(*other.p_vector, std::move(scalar));
        return instance();
    }
    Derived& sub_scal_mul(const AlgebraBase& other, ScalarCRef scalar)
    {
        p_vector->sub_scal_mul(*other.p_vector, std::move(scalar));
        return instance();
    }

    Derived& add_scal_div(const AlgebraBase& other, ScalarCRef scalar)
    {
        const auto recip = devices::math::reciprocal(scalar);
        return add_scal_mul(other, recip);
    }
    Derived& sub_scal_div(const AlgebraBase& other, ScalarCRef scalar)
    {
        const auto recip = devices::math::reciprocal(scalar);
        return add_scal_mul(other, recip);
    }

    RPY_NO_DISCARD bool is_equal(const AlgebraBase& other) const noexcept
    {
        return p_vector->is_equal(*other.p_vector);
    }

    /**
     * @brief Multiply the algebra by another vector on the right.
     *
     * Multiply *this on the right by other result = (*this) * other;
     *
     * @param other The vector to multiply with.
     * @return Algebra The result of the multiplication.
     */
    RPY_NO_DISCARD Derived right_multiply(const Vector& other) const
    {
        auto result = Derived::new_like(*this);
        result.fma(other, *this, ops::IdentityOperator{});
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
    RPY_NO_DISCARD Derived left_multiply(const Vector& other) const
    {
        auto result = Derived::new_like(*this);
        result.fma(other, *this, ops::IdentityOperator{});
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
    Derived& multiply_inplace(const Vector& other)
    {
        instance().multiply_inplace(other, ops::IdentityOperator{});
        return instance();
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
    Derived& add_multiply(const Vector& left, const Vector& right)
    {
        instance().fma(left, right, ops::IdentityOperator{});
        return instance();
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
    Derived& sub_multiply(const Vector& left, const Vector& right)
    {
        instance().fma(left, right, ops::UnaryMinusOperator{});
        return instance();
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
    Derived& multiply_post_smul(const Vector& rhs, ScalarCRef multiplier)
    {
        instance().multiply_inplace(
                rhs,
                ops::RightMultiplyOperator(std::move(multiplier))
        );
        return instance();
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
    Derived& multiply_post_sdiv(const Vector& rhs, ScalarCRef divisor)
    {
        auto recip = devices::math::reciprocal(divisor);
        return multiply_post_smul(rhs, recip);
    }
};

template <typename Derived>
inline constexpr bool is_algebra_v
        = is_base_of_v<AlgebraBase<Derived>, Derived>;

namespace dtl {

template <typename Return, bool AndCondition = true>
using algebra_like_return = enable_if_t<
        is_algebra_v<remove_cv_ref_t<Return>> && AndCondition,
        Return>;

}

template <typename Derived>
struct VectorTraits<Derived, enable_if_t<is_algebra_v<Derived>>> {
    static constexpr bool is_vector = true;

    static Vector& as_mut_vector(Derived& arg) noexcept
    {
        return arg.as_vector();
    }

    static const Vector& as_vector(const Derived& arg) noexcept
    {
        return arg.as_vector();
    }

    static Derived clone(const Derived& arg) noexcept
    {
        return Derived::clone(arg);
    }

    static Derived new_like(const Derived& arg) noexcept
    {
        return Derived::new_like(arg.as_vector());
    }

    static Rc<Vector> new_ptr_like(const Derived& arg) noexcept
    {
        return new Derived(new_like(arg));
    }

    static Derived from_like(const Derived& RPY_UNUSED_VAR like, Vector&& arg)
    {
        return Derived(new Vector(std::move(arg)));
    }
};

/**
 * @brief Abstract base class for implementing multiplication operations on
 * vectors.
 *
 * The Multiplication class provides a common interface for various
 * multiplication operations that can be performed on vectors using certain
 * operators.
 */
class Multiplication : public RcBase<Multiplication>
{
    // NOLINTBEGIN: readability-identifier-length
public:
    virtual ~Multiplication() = default;

    virtual void fma(Vector& a, const Vector& b, const Vector& c, const ops::Operator& op
    ) const = 0;

    virtual void
    inplace_multiply(Vector& lhs, const Vector& rhs, const ops::Operator& op)
            const
            = 0;
    // NOLINTEND
};


using MultiplicationPtr = Rc<const Multiplication>;

/**
 * @class Algebra
 * @brief Represents algebraic structures and operations.
 *
 * This class encapsulates various algebraic operations and provides methods
 * to perform calculations commonly used in algebra.
 */
class Algebra : public AlgebraBase<Algebra>
{
    MultiplicationPtr p_multiplication;

public:
    static Algebra clone(const Algebra& other) noexcept;
    static Algebra new_like(const Algebra& other) noexcept;
    static Algebra from(Vector&& arg) noexcept;

    Algebra(BasisPointer basis,
            scalars::TypePtr scalar_type,
            MultiplicationPtr multiplication) noexcept;
    Algebra(Vector&& data, MultiplicationPtr multiplication) noexcept;

    RPY_NO_DISCARD const Rc<const Multiplication>&
    multiplication() const noexcept
    {
        return p_multiplication;
    }

    // NOLINTNEXTLINE: readability-identifier-length
    Algebra& fma(const Vector& lhs, const Vector& rhs, const ops::Operator& op)
    {
        p_multiplication->fma(as_vector(), lhs, rhs, op);
        return *this;
    }

    // NOLINTNEXTLINE: readability-identifier-length
    Algebra& inplace_multiply(const Vector& rhs, const ops::Operator& op)
    {
        p_multiplication->inplace_multiply(as_vector(), rhs, op);
        return *this;
    }
};

template <>
struct VectorTraits<Algebra, void> {
    static constexpr bool is_vector = true;

    static Vector& as_mut_vector(Algebra& arg) noexcept
    {
        return arg.as_vector();
    }

    static const Vector& as_vector(const Algebra& arg) noexcept
    {
        return arg.as_vector();
    }

    static Algebra clone(const Algebra& arg) noexcept { return {arg}; }

    static Algebra new_like(const Algebra& arg) noexcept
    {
        return {arg.basis(), arg.scalar_type(), arg.multiplication()};
    }

    static Algebra from_like(const Algebra& like, Vector&& arg)
    {
        RPY_CHECK(like.basis() == arg.basis());

        return {std::move(arg), like.multiplication()};
    }
};

/******************************************************************************
 *  Implementations of all the free-standing operators for algebras.          *
 ******************************************************************************/

template <typename Alg, typename Vec>
RPY_NO_DISCARD enable_if_t<is_algebra_v<Alg> && is_vector_v<Vec>, Alg>
operator*(const Alg& lhs, const Vec& rhs)
{
    return lhs.right_multiply(VectorTraits<Vec>::as_vector(rhs));
    ;
}

template <typename Alg, typename Vec>
RPY_NO_DISCARD enable_if_t<
        is_algebra_v<Alg> && is_vector_v<Vec> && !is_algebra_v<Vec>,
        Alg>
operator*(const Vec& lhs, const Alg& rhs)
{
    return rhs.left_multiply(VectorTraits<Vec>::as_vector(lhs));
    ;
}

template <typename Alg, typename Vec>
RPY_NO_DISCARD enable_if_t<is_algebra_v<Alg> && is_vector_v<Vec>, Alg&>
operator*=(Alg& lhs, const Vec& rhs)
{
    lhs.right_multiply_inplace(VectorTraits<Vec>::as_vector(rhs));
    return lhs;
}

// NOLINTBEGIN: readability-identifier-length
template <typename Alg, typename Vec1, typename Vec2>
RPY_NO_DISCARD enable_if_t<
        is_algebra_v<Alg> && is_vector_v<Vec1> && is_vector_v<Vec2>,
        Alg>
fused_multiply_add(const Alg& a, const Vec1& b, const Vec2& c)
{
    auto result = Alg::clone(a);
    result.fma(b, c);
    return result;
}

template <typename Alg, typename Vec1, typename Vec2>
enable_if_t<is_algebra_v<Alg> && is_vector_v<Vec1> && is_vector_v<Vec2>, Alg&>
inplace_fused_multiply_add(Alg& a, const Vec1& b, const Vec2& c)
{
    a.fma(b, c);
    return a;
}

template <typename Alg, typename Vec>
enable_if_t<is_algebra_v<Alg> && is_vector_v<Vec>, Alg>
commutator(const Alg& a, const Vec& b)
{
    auto result = a.right_multiply(VectorTraits<Vec>::as_vector(b));
    inplcae_fused_multiply_add(result, b, a);
    return result;
}

// NOLINTEND


}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_ALGEBRA_H
