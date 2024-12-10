//
// Created by sammorley on 15/11/24.
//

#include "roughpy/generics/number_trait.h"

#include <stdexcept>

#include "roughpy/core/check.h"
#include "roughpy/core/debug_assertion.h"

#include "roughpy/generics/type.h"
#include "roughpy/generics/values.h"

using namespace rpy;
using namespace rpy::generics;

NumberTrait::~NumberTrait() = default;

void NumberTrait::unsafe_real(void* dst, const void* src) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    p_type->copy_or_fill(dst, src, 1, false);
}
void NumberTrait::unsafe_imaginary(void* dst, const void* src) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    p_type->copy_or_fill(dst, nullptr, 1, false);
}
void NumberTrait::unsafe_sqrt(void* dst, const void* src) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_THROW(std::runtime_error, "this operation is not implemented");
}
void NumberTrait::unsafe_pow(void* dst, const void* base, exponent_t exponent)
        const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_THROW(std::runtime_error, "this operation is not implemented");
}
void NumberTrait::unsafe_exp(void* dst, const void* src) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_THROW(std::runtime_error, "this operation is not implemented");
}
void NumberTrait::unsafe_log(void* dst, const void* src) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_THROW(std::runtime_error, "this operation is not implemented");
}

void NumberTrait::real(Ref dst, ConstRef src) const
{
    RPY_CHECK(dst.is_valid() && src.is_valid());
    RPY_CHECK_EQ(dst.type(), *real_type());
    RPY_CHECK_EQ(src.type(), *p_type);

    unsafe_real(dst.data(), src.data());
}
void NumberTrait::imaginary(Ref dst, ConstRef src) const
{
    RPY_CHECK(imaginary_type(), "this type is not complex");
    RPY_CHECK(dst.is_valid() && src.is_valid());
    RPY_CHECK_EQ(dst.type(), *real_type());
    RPY_CHECK_EQ(src.type(), *p_type);

    unsafe_imaginary(dst.data(), src.data());
}
void NumberTrait::abs(Ref dst, ConstRef src) const
{
    RPY_CHECK(dst.is_valid() && src.is_valid());
    RPY_CHECK_EQ(dst.type(), *real_type());
    RPY_CHECK_EQ(src.type(), *p_type);

    unsafe_abs(dst.data(), src.data());
}
void NumberTrait::sqrt(Ref dst, ConstRef src) const
{
    RPY_CHECK(dst.is_valid() && src.is_valid());
    RPY_CHECK_EQ(dst.type(), *real_type());
    RPY_CHECK_EQ(src.type(), *p_type);

    unsafe_abs(dst.data(), src.data());
}
void NumberTrait::exp(Ref dst, ConstRef src) const
{
    RPY_CHECK(dst.is_valid() && src.is_valid());
    RPY_CHECK_EQ(dst.type(), *p_type);
    RPY_CHECK_EQ(src.type(), *p_type);

    unsafe_exp(dst.data(), src.data());
}
void NumberTrait::log(Ref dst, ConstRef src) const
{
    RPY_CHECK(dst.is_valid() && src.is_valid());
    RPY_CHECK_EQ(dst.type(), *p_type);
    RPY_CHECK_EQ(src.type(), *p_type);

    unsafe_log(dst.data(), src.data());
}
void NumberTrait::from_rational(
        Ref dst,
        int64_t numerator,
        int64_t denominator
) const
{
    RPY_CHECK_NE(denominator, 0, std::domain_error);

    RPY_CHECK(!dst.fast_is_zero());
    RPY_CHECK_EQ(dst.type(), *p_type);

    unsafe_from_rational(dst.data(), numerator, denominator);
}
Value NumberTrait::real(ConstRef value) const
{
    RPY_CHECK(value.is_valid());
    RPY_CHECK_EQ(value.type(), *p_type);

    Value result(p_type);
    unsafe_real(result.data(), value.data());

    return result;
}
Value NumberTrait::imaginary(ConstRef value) const
{
    RPY_CHECK(value.is_valid());
    RPY_CHECK_EQ(value.type(), *p_type);

    Value result(p_type);
    unsafe_real(result.data(), value.data());

    return result;
}
Value NumberTrait::abs(ConstRef value) const
{
    RPY_CHECK(value.is_valid());
    RPY_CHECK_EQ(value.type(), *p_type);

    Value result(real_type());
    unsafe_abs(result.data(), value.data());

    return result;
}
Value NumberTrait::sqrt(ConstRef value) const
{
    RPY_CHECK(value.is_valid());
    RPY_CHECK_EQ(value.type(), *p_type);

    Value result(p_type);
    unsafe_sqrt(result.data(), value.data());

    return result;
}
Value NumberTrait::pow(ConstRef value, exponent_t power) const
{
    RPY_CHECK(value.is_valid());
    RPY_CHECK_EQ(value.type(), *p_type);

    Value result(p_type);
    if (power != 1) {
        unsafe_pow(result.data(), value.data(), power);
    } else {
        p_type->copy_or_fill(result.data(), value.data(), 1, false);
    }

    return result;
}
Value NumberTrait::exp(ConstRef value) const
{
    RPY_CHECK(value.is_valid());
    RPY_CHECK_EQ(value.type(), *p_type);

    Value result(p_type);
    unsafe_exp(result.data(), value.data());

    return result;
}
Value NumberTrait::log(ConstRef value) const
{
    RPY_CHECK(value.is_valid());
    RPY_CHECK_EQ(value.type(), *p_type);

    Value result(p_type);
    unsafe_exp(result.data(), value.data());

    return result;
}
Value NumberTrait::from_rational(int64_t numerator, int64_t denominator) const
{
    RPY_CHECK_NE(denominator, 0, std::domain_error);

    Value result(p_type);
    unsafe_from_rational(result.data(), numerator, denominator);

    return result;
}
