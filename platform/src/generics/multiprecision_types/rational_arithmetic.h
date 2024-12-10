//
// Created by sammorley on 25/11/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_RATIONAL_ARITHMETIC_H
#define ROUGHPY_GENERICS_INTERNAL_RATIONAL_ARITHMETIC_H

#include "roughpy/generics/arithmetic_trait.h"

namespace rpy {
namespace generics {

class RationalArithmetic : public ArithmeticTrait {

public:

    explicit RationalArithmetic(const Type* type)
        : ArithmeticTrait(type, type)
    {}

    RPY_NO_DISCARD bool
    has_operation(ArithmeticOperation op) const noexcept override;
    void unsafe_add_inplace(void* lhs, const void* rhs) const noexcept override;
    void unsafe_sub_inplace(void* lhs, const void* rhs) const override;
    void unsafe_mul_inplace(void* lhs, const void* rhs) const override;
    void unsafe_div_inplace(void* lhs, const void* rhs) const override;
};

} // generics
} // rpy

#endif //ROUGHPY_GENERICS_INTERNAL_RATIONAL_ARITHMETIC_H
