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


namespace rpy {
namespace algebra {


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


public:
    using Scalar = scalars::Scalar;
    using ScalarCRef = scalars::ScalarCRef;
    using const_iterator = Vector::const_iterator;

protected:
    Derived& instance() noexcept { return dtl::cast_algebra_to_derived(*this); }
    const Derived& instance() const noexcept
    {
        return dtl::cast_algebra_to_derived(*this);
    }

    const typename Derived::multiplication_t& multiplication() const noexcept
    {
        return instance().multiplication();
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

    RPY_NO_DISCARD Derived minus() const { return p_vector->uminus(); }
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
    RPY_NO_DISCARD Derived right_multiply(const Vector& other) const;

    /**
     * @brief Multiply the algebra by another vector on left.
     *
     * Multiply *this on the left by other result = other * (*this).
     *
     * @param other The vector to multiply with.
     * @return Algebra The result of the multiplication.
     */
    RPY_NO_DISCARD Derived left_multiply(const Vector& other) const;

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
    Derived& multiply_inplace(const Vector& other);

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
    Derived& add_multiply(const Vector& left, const Vector& right);

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
    Derived& sub_multiply(const Vector& left, const Vector& right);

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
    Derived& multiply_post_smul(const Vector& rhs, ScalarCRef multiplier);

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
    Derived& multiply_post_sdiv(const Vector& rhs, ScalarCRef divisor);
};


template <typename Derived>
inline constexpr bool is_algebra_v
        = is_base_of_v<AlgebraBase<Derived>, Derived>;

namespace dtl {

template <typename Return, bool AndCondition=true>
using algebra_like_return
        = enable_if_t<is_algebra_v<remove_cv_ref_t<Return>> && AndCondition, Return>;

}

template <typename Derived>
struct VectorTraits<Derived, enable_if_t<is_algebra_v<Derived>>>
{
    static constexpr bool is_vector = true;

    static Vector& as_mut_vector(Derived& arg) noexcept
    {
        return arg.as_vector();
    }

    static const Vector& as_vector(const Derived& arg) noexcept
    {
        return arg.as_vector();
    }

    static Derived new_like(const Derived& arg) noexcept
    {
        return Derived::new_like(arg.as_vector());
    }

    static Rc<Vector> new_ptr_like(const Derived& arg) noexcept
    {
        return new Derived(new_like(arg));
    }

    static Derived from(Vector&& arg)
    {
        return Derived(new Vector(std::move(arg)));
    }


};

template <typename Derived>
RPY_NO_DISCARD dtl::algebra_like_return<Derived> operator-(const Derived& arg)
{
    return arg.minus();
}

template <typename Derived>
RPY_NO_DISCARD dtl::algebra_like_return<Derived>
operator*(const Derived& vec, scalars::ScalarCRef scalar)
{
    return vec.right_smul(std::move(scalar));
}

template <typename Derived>
RPY_NO_DISCARD dtl::algebra_like_return<Derived>
operator*(scalars::ScalarCRef scalar, const Derived& vec)
{
    return vec.left_smul(std::move(scalar));
}

template <typename Derived>
RPY_NO_DISCARD dtl::algebra_like_return<Derived>
operator/(const Derived& vec, scalars::ScalarCRef scalar)
{
    const auto recip = devices::math::reciprocal(scalar);
    return vec.right_smul(recip);
}

template <typename Derived>
RPY_NO_DISCARD dtl::algebra_like_return<Derived>
operator+(const Derived& lhs, const Vector& rhs)
{
    return Derived::from(lhs.as_vector().add(rhs));
}

template <typename Derived>
RPY_NO_DISCARD dtl::algebra_like_return<Derived>
operator+(const Vector& lhs, const Derived& rhs)
{
    return Derived::from(lhs.add(rhs.as_vector()));
}

template <typename Derived>
RPY_NO_DISCARD dtl::algebra_like_return<Derived>
operator-(const Derived& lhs, const Vector& rhs)
{
    return Derived::from(lhs.as_vector().sub(rhs));
}

template <typename Derived>
RPY_NO_DISCARD dtl::algebra_like_return<Derived>
operator-(const Vector& lhs, const Derived& rhs)
{
    return Derived::from(lhs.sub(rhs.as_vector()));
}

template <typename Derived, typename OtherDerived>
RPY_NO_DISCARD dtl::algebra_like_return<Derived>
operator+(const Derived& lhs, const AlgebraBase<OtherDerived>& rhs)
{
    return Derived::from(lhs.as_vector().add(rhs.as_vector()));
}

template <typename Derived, typename OtherDerived>
RPY_NO_DISCARD dtl::algebra_like_return<Derived>
operator-(const Derived& lhs, const AlgebraBase<OtherDerived>& rhs)
{
    return Derived::from(lhs.as_vector().sub(rhs.as_vector()));
}

template <typename Derived>
dtl::algebra_like_return<Derived&>
operator*=(Derived& lhs, scalars::ScalarCRef scalar)
{
    return lhs.right_smul_inplace(scalar);
}

template <typename Derived>
dtl::algebra_like_return<Derived&>
operator/=(Derived& vec, scalars::ScalarCRef scalar)
{
    const auto recip = devices::math::reciprocal(scalar);
    return vec.right_smul_inplace(recip);
}

template <typename Derived>
dtl::algebra_like_return<Derived&> operator+=(Derived& lhs, const Vector& rhs)
{
    lhs.as_vector().add_inplace(rhs);
    return lhs;
}

template <typename Derived, typename OtherDerived>
dtl::algebra_like_return<Derived&>
operator+=(Derived& lhs, const AlgebraBase<OtherDerived>& rhs)
{
    lhs.add_inplace(rhs);
    return lhs;
}

template <typename Derived>
dtl::algebra_like_return<Derived&>
operator-=(AlgebraBase<Derived>& lhs, const Vector& rhs)
{
    lhs.as_vector().subinplace(rhs);
    return lhs;
}

template <typename Derived, typename OtherDerived>
dtl::algebra_like_return<Derived&>
operator-=(Derived& lhs, const AlgebraBase<OtherDerived>& rhs)
{
    lhs.sub_inplace(rhs);
    return lhs;
}

template <typename Derived, typename OtherDerived>
RPY_NO_DISCARD dtl::algebra_like_return<Derived> add_scalar_multiply(
        const Derived& lhs,
        const AlgebraBase<OtherDerived>& rhs,
        scalars::ScalarCRef scalar
)
{
    auto result = Derived::new_like(lhs);
    result.as_vector().add_scal_mul(rhs.as_vector(), std::move(scalar));
    return result;
}

template <typename Derived, typename OtherDerived>
RPY_NO_DISCARD dtl::algebra_like_return<Derived> add_scalar_multiply(
        const Derived& lhs,
        scalars::ScalarCRef scalar,
        const AlgebraBase<OtherDerived>& rhs
)
{
    auto result = Derived::new_like(lhs);
    result.as_vector().add_scal_mul(rhs.as_vector(), std::move(scalar));
    return result;
}

template <typename Derived, typename OtherDerived>
RPY_NO_DISCARD dtl::algebra_like_return<Derived> sub_scalar_multiply(
        const Derived& lhs,
        const AlgebraBase<OtherDerived>& rhs,
        scalars::ScalarCRef scalar
)
{
    auto result = Derived::new_like(lhs);
    result.as_vector().sub_scal_mul(rhs.as_vector(), std::move(scalar));
    return result;
}

template <typename Derived, typename OtherDerived>
RPY_NO_DISCARD dtl::algebra_like_return<Derived> sub_scalar_multiply(
        const Derived& lhs,
        scalars::ScalarCRef scalar,
        const AlgebraBase<OtherDerived>& rhs
)
{
    auto result = Derived::new_like(lhs);
    result.as_vector().sub_scal_mul(rhs.as_vector(), std::move(scalar));
    return result;
}

template <typename Derived>
RPY_NO_DISCARD dtl::algebra_like_return<Derived> add_scalar_multiply(
        const Derived& lhs,
        const Vector& rhs,
        scalars::ScalarCRef scalar
)
{
    auto result = Derived::new_like(lhs);
    result.as_vector().add_scal_mul(rhs, std::move(scalar));
    return result;
}

template <typename Derived>
RPY_NO_DISCARD dtl::algebra_like_return<Derived> add_scalar_multiply(
        const Derived& lhs,
        scalars::ScalarCRef scalar,
        const Vector& rhs
)
{
    auto result = Derived::new_like(lhs);
    result.as_vector().add_scal_mul(rhs, std::move(scalar));
    return result;
}

template <typename Derived>
RPY_NO_DISCARD dtl::algebra_like_return<Derived> sub_scalar_multiply(
        const Derived& lhs,
        const Vector& rhs,
        scalars::ScalarCRef scalar
)
{
    auto result = Derived::new_like(lhs);
    result.as_vector().sub_scal_mul(rhs, std::move(scalar));
    return result;
}

template <typename Derived>
RPY_NO_DISCARD dtl::algebra_like_return<Derived> sub_scalar_multiply(
        const Derived& lhs,
        scalars::ScalarCRef scalar,
        const Vector& rhs
)
{
    auto result = Derived::new_like(lhs);
    result.as_vector().sub_scal_mul(rhs, std::move(scalar));
    return result;
}




}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_ALGEBRA_H
