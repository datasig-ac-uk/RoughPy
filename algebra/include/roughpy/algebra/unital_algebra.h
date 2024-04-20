//
// Created by sam on 4/19/24.
//

#ifndef ROUGHPY_ALGEBRA_UNITAL_ALGEBRA_H
#define ROUGHPY_ALGEBRA_UNITAL_ALGEBRA_H

#include "algebra.h"

namespace rpy {
namespace algebra {

namespace dtl {
template <typename Multiplication>
struct UnitalAlgebraTraits {
    static BasisKey unit_key(const BasisPointer& basis) noexcept;
};
}// namespace dtl

/**
 * @class UnitalAlgebra
 * @brief Represents a unital algebra.
 */
template <typename Multiplication, typename Base = Algebra<Multiplication>>
class UnitalAlgebra : public Base
{
    using unital_traits = dtl::UnitalAlgebraTraits<Multiplication>;

protected:
    static bool basis_compatibility_check(const Basis& basis) noexcept
    {
        return basis.has_key(unital_traits::unit_key())
                && Base::basis_compatibility_check(basis);
    }

public:
    using typename Base::Scalar;

    using Base::Base;

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

template <typename Multiplication>
using GradedUnitalAlgebra
        = UnitalAlgebra<Multiplication, GradedAlgebra<Multiplication>>;

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_UNITAL_ALGEBRA_H
