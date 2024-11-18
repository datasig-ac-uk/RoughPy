//
// Created by sammorley on 18/11/24.
//

#include "roughpy/generics/values.h"

#include "roughpy/generics/builtin_trait.h"
#include "roughpy/generics/number_trait.h"

using namespace rpy;
using namespace rpy::generics;

namespace {


RPY_NO_DISCARD
inline Value abs_impl(const NumberTrait* trait, const Type* type, const void*
data)
{
    Value result(TypePtr(trait->real_type()));
    trait->unsafe_abs(result.data(), data);
    return result;
}

RPY_NO_DISCARD
inline Value sqrt_impl(const NumberTrait* trait, const Type* tp, const void*
data)
{
    Value result(tp);
    trait->unsafe_sqrt(result.data(), data);
    return result;
}

RPY_NO_DISCARD inline Value pow_impl(
        const NumberTrait* trait,
        const Type* tp,
        const void* data
)
{
    using payload_t = std::pair<const void*, exponent_t>;
    Value result(tp);
    const auto& [base_ptr, exp] = *static_cast<const payload_t*>(data);
    trait->unsafe_pow(result.data(), base_ptr, exp);
    return result;
}

RPY_NO_DISCARD inline Value exp_impl(const NumberTrait* trait, const Type* tp,
const void* data)
{
    Value result(tp);
    trait->unsafe_exp(result.data(), data);
    return result;
}

RPY_NO_DISCARD inline Value log_impl(const NumberTrait* trait, const Type* 
tp, const void* data)
{
    Value result(tp);
    trait->unsafe_log(result.data(), data);
    return result;
}

RPY_NO_DISCARD inline Value real_impl(const NumberTrait* trait, const Type* tp, 
const void* data)
{
    Value result(trait->real_type());
    trait->unsafe_real(result.data(), data);
    return result;
}

RPY_NO_DISCARD inline Value imag_impl(const NumberTrait* trait, const Type* 
tp, const void* data)
{
    Value result(trait->real_type());
    trait->unsafe_imaginary(result.data(), data);
    return result;
}


}

Value generics::dtl::math_fn(
        NumberFunction func,
        const Type* type,
        const void* data
)
{
    RPY_DBG_ASSERT_NE(type, nullptr);
    RPY_CHECK_NE(data, nullptr);
    const auto* trait = trait_cast<NumberTrait>(
            type->get_builtin_trait(BuiltinTraitID::Number)
    );

    RPY_CHECK_NE(trait, nullptr);
    RPY_CHECK(trait->has_function(func));

    switch (func) {
        case NumberFunction::Abs: return abs_impl(trait, type, data);
        case NumberFunction::Sqrt: return sqrt_impl(trait, type, data);
        case NumberFunction::Pow: return pow_impl(trait, type, data);
        case NumberFunction::Exp: return exp_impl(trait, type, data);
        case NumberFunction::Log: return log_impl(trait, type, data);
        case NumberFunction::Real: return real_impl(trait, type, data);
        case NumberFunction::Imaginary: return imag_impl(trait, type, data);
        default: break;
    }

    RPY_THROW(std::runtime_error, "unsupported operation");
}


void generics::dtl::from_rational(
        const Type* type,
        void* dst,
        int64_t numerator,
        int64_t denominator
)
{
    RPY_CHECK_NE(denominator, 0, std::domain_error);
    RPY_DBG_ASSERT_NE(type, nullptr);
    RPY_CHECK_NE(dst, nullptr);
    const auto* trait = trait_cast<NumberTrait>(
            type->get_builtin_trait(BuiltinTraitID::Number)
    );

    RPY_CHECK_NE(trait, nullptr);
    RPY_CHECK(trait->has_function(NumberFunction::FromRational));

    trait->unsafe_from_rational(dst, numerator, denominator);
}
