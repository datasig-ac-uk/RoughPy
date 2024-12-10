//
// Created by sam on 27/11/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_POLYNOMIAL_ARITHMETIC_H
#define ROUGHPY_GENERICS_INTERNAL_POLYNOMIAL_ARITHMETIC_H

#include "roughpy/core/macros.h"

#include "roughpy/generics/arithmetic_trait.h"


namespace rpy {
namespace generics {

class PolynomialArithmetic : public ArithmeticTrait {

public:

    explicit PolynomialArithmetic(const Type* type, const Type* rational_type)
        : ArithmeticTrait(type, rational_type)
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

#endif //ROUGHPY_GENERICS_INTERNAL_POLYNOMIAL_ARITHMETIC_H
