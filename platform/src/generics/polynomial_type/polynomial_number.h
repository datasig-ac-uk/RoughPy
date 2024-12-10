//
// Created by sam on 27/11/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_POLYNOMIAL_NUMBER_H
#define ROUGHPY_GENERICS_INTERNAL_POLYNOMIAL_NUMBER_H

#include "roughpy/generics/number_trait.h"

namespace rpy {
namespace generics {

class PolynomialNumber : public NumberTrait {

public:
    explicit PolynomialNumber(const Type* tp) noexcept
        : NumberTrait(tp, nullptr)
    {}

    bool has_function(NumberFunction fn_id) const noexcept override;

    void unsafe_abs(void* dst, const void* src) const noexcept override;

    void unsafe_from_rational(void* dst,
        int64_t numerator,
        int64_t denominator) const override;
};

} // generics
} // rpy

#endif //ROUGHPY_GENERICS_INTERNAL_POLYNOMIAL_NUMBER_H