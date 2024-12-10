//
// Created by sammorley on 25/11/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_FLOAT_NUMBER_H
#define ROUGHPY_GENERICS_INTERNAL_FLOAT_NUMBER_H


#include "roughpy/generics/number_trait.h"

namespace rpy {
namespace generics {

class FloatNumber : public NumberTrait {
public:

    explicit FloatNumber(const Type* type) : NumberTrait(type, type) {}

    bool has_function(NumberFunction fn_id) const noexcept override;

    void unsafe_real(void* dst, const void* src) const override;

    void unsafe_imaginary(void* dst, const void* src) const override;

    void unsafe_abs(void* dst, const void* src) const noexcept override;

    void unsafe_sqrt(void* dst, const void* src) const override;

    void unsafe_pow(void* dst,
        const void* base,
        exponent_t exponent) const override;

    void unsafe_exp(void* dst, const void* src) const override;

    void unsafe_log(void* dst, const void* src) const override;

    void unsafe_from_rational(void* dst,
        int64_t numerator,
        int64_t denominator) const override;
};

} // generics
} // rpy

#endif //ROUGHPY_GENERICS_INTERNAL_FLOAT_NUMBER_H
