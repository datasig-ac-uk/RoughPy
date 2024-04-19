//
// Created by sam on 1/29/24.
//

#ifndef ROUGHPY_ALGEBRA_ALGEBRA_H
#define ROUGHPY_ALGEBRA_ALGEBRA_H

#include "roughpy_algebra_export.h"
#include "vector.h"

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

namespace rpy {
namespace algebra {

namespace dtl {

template <typename Multiplication>
struct MultiplicationTraits {
    static void fma(Vector& out, const Vector& left, const Vector& right);
    static void multiply_into(Vector& out, const Vector& other);
};

template <typename Multiplication>
struct UnitalAlgebraTraits {
    static BasisKey unit_key(const BasisPointer& basis) noexcept;
};

}// namespace dtl

template <typename Multiplication, typename Base = Vector>
class Algebra : public Base
{
    using multiplication_type = Multiplication;
    using multiplication_traits = dtl::MultiplicationTraits<Multiplication>;

public:
    RPY_NO_DISCARD Algebra multiply(const Vector& other) const;
    Algebra& multiply_inplace(const Vector& other);
};

template <typename Multiplication, typename Base>
Algebra<Multiplication, Base>
Algebra<Multiplication, Base>::multiply(const Vector& other) const
{
    Algebra result(this->basis());
    multiplication_traits::fma(result, *this, other);
    return result;
}

template <typename Multiplication, typename Base>
Algebra<Multiplication, Base>&
Algebra<Multiplication, Base>::multiply_inplace(const Vector& other)
{
    multiplication_traits::multiply_into(*this, other);
    return *this;
}

/**
 * @class UnitalAlgebra
 * @brief represents a unital algebra.
 */
template <typename Multiplication, typename Base = Algebra<Multiplication>>
class UnitalAlgebra : public Base
{
    using unital_traits = dtl::UnitalAlgebraTraits<Multiplication>;

public:
    using typename Base::Scalar;

    UnitalAlgebra(BasisPointer basis, Scalar unit_coeff)
        : Base(basis, unital_traits::unit_key(basis), std::move(unit_coeff))
    {}

    /**
     * @brief Get the coefficient of the unit of the algebra.
     *
     * @return Scalar The coefficient of the unit of the algebra.
     */
    Scalar unit() const
    {
        return (*this)[unital_traits::unit_key(this->basis())];
    }
};

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_ALGEBRA_H
