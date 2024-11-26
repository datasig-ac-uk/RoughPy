//
// Created by sammorley on 25/11/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_INTEGER_COMPARISON_H
#define ROUGHPY_GENERICS_INTERNAL_INTEGER_COMPARISON_H

#include "roughpy/generics/comparison_trait.h"

namespace rpy {
namespace generics {

class IntegerComparison : public ComparisonTrait
{
public:
    explicit IntegerComparison(const Type* type) : ComparisonTrait(type) {}

    RPY_NO_DISCARD bool
    has_comparison(ComparisonType comp) const noexcept override;
    RPY_NO_DISCARD bool unsafe_compare_equal(
            const void* lhs,
            const void* rhs
    ) const noexcept override;
    RPY_NO_DISCARD bool unsafe_compare_less(
            const void* lhs,
            const void* rhs
    ) const noexcept override;
    RPY_NO_DISCARD bool unsafe_compare_less_equal(
            const void* lhs,
            const void* rhs
    ) const noexcept override;
    RPY_NO_DISCARD bool unsafe_compare_greater(
            const void* lhs,
            const void* rhs
    ) const noexcept override;
    RPY_NO_DISCARD bool unsafe_compare_greater_equal(
            const void* lhs,
            const void* rhs
    ) const noexcept override;
};

}// namespace generics
}// namespace rpy

#endif// ROUGHPY_GENERICS_INTERNAL_INTEGER_COMPARISON_H
