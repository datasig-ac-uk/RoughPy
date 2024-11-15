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

NumberTraits::~NumberTraits() = default;

void NumberTraits::unsafe_real(void* dst, const void* src) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    p_type->copy(dst, src, 1, true);
}
void NumberTraits::unsafe_imaginary(void* dst, const void* src) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    p_type->copy(dst, nullptr, 1, false);
}
void NumberTraits::unsafe_sqrt(void* dst, const void* src) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_THROW(std::runtime_error, "this operation is not implemented");
}
void NumberTraits::unsafe_pow(void* dst, const void* base, int64_t exponent)
        const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_THROW(std::runtime_error, "this operation is not implemented");
}
void NumberTraits::unsafe_exp(void* dst, const void* src) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_THROW(std::runtime_error, "this operation is not implemented");
}
void NumberTraits::unsafe_log(void* dst, const void* src) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_THROW(std::runtime_error, "this operation is not implemented");
}

void NumberTraits::real(Ref dst, ConstRef src) const
{
    RPY_CHECK(dst.is_valid() && src.is_valid());
    RPY_CHECK_EQ(dst.type(), *real_type());
    RPY_CHECK_EQ(src.type(), *p_type);

    unsafe_real(dst.data(), src.data());
}
void NumberTraits::imaginary(Ref dst, ConstRef src) const
{
    RPY_CHECK(imaginary_type(), "this type is not complex");
    RPY_CHECK(dst.is_valid() && src.is_valid());
    RPY_CHECK_EQ(dst.type(), *real_type());
    RPY_CHECK_EQ(src.type(), *p_type);

    unsafe_imaginary(dst.data(), src.data());
}
void NumberTraits::abs(Ref dst, ConstRef src) const
{
    RPY_CHECK(dst.is_valid() && src.is_valid());
    RPY_CHECK_EQ(dst.type(), *real_type());
    RPY_CHECK_EQ(src.type(), *p_type);

    unsafe_abs(dst.data(), src.data());
}
void NumberTraits::sqrt(Ref dst, ConstRef src) const
{
    RPY_CHECK(dst.is_valid() && src.is_valid());
    RPY_CHECK_EQ(dst.type(), *real_type());
    RPY_CHECK_EQ(src.type(), *p_type);

    unsafe_abs(dst.data(), src.data());
}
void NumberTraits::exp(Ref dst, ConstRef src) const
{
    RPY_CHECK(dst.is_valid() && src.is_valid());
    RPY_CHECK_EQ(dst.type(), *p_type);
    RPY_CHECK_EQ(src.type(), *p_type);

    unsafe_exp(dst.data(), src.data());
}
void NumberTraits::log(Ref dst, ConstRef src) const
{
    RPY_CHECK(dst.is_valid() && src.is_valid());
    RPY_CHECK_EQ(dst.type(), *p_type);
    RPY_CHECK_EQ(src.type(), *p_type);

    unsafe_log(dst.data(), src.data());
}
void NumberTraits::from_rational(
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
Value NumberTraits::real(ConstRef value) const
{
    RPY_CHECK(value.is_valid());
    RPY_CHECK_EQ(value.type(), *p_type);

    Value result(p_type);
    unsafe_real(result.data(), value.data());

    return result;
}
Value NumberTraits::imaginary(ConstRef value) const
{
    RPY_CHECK(value.is_valid());
    RPY_CHECK_EQ(value.type(), *p_type);

    Value result(p_type);
    unsafe_real(result.data(), value.data());

    return result;
}
Value NumberTraits::abs(ConstRef value) const
{
    RPY_CHECK(value.is_valid());
    RPY_CHECK_EQ(value.type(), *p_type);

    Value result(real_type());
    unsafe_abs(result.data(), value.data());

    return result;
}
Value NumberTraits::sqrt(ConstRef value) const
{
    RPY_CHECK(value.is_valid());
    RPY_CHECK_EQ(value.type(), *p_type);

    Value result(p_type);
    unsafe_sqrt(result.data(), value.data());

    return result;
}
Value NumberTraits::pow(ConstRef value, int64_t power) const
{
    RPY_CHECK(value.is_valid());
    RPY_CHECK_EQ(value.type(), *p_type);

    Value result(p_type);
    if (power != 1) {
        unsafe_pow(result.data(), value.data(), power);
    } else {
        p_type->copy(result.data(), value.data(), 1, true);
    }

    return result;
}
Value NumberTraits::exp(ConstRef value) const
{
    RPY_CHECK(value.is_valid());
    RPY_CHECK_EQ(value.type(), *p_type);

    Value result(p_type);
    unsafe_exp(result.data(), value.data());

    return result;
}
Value NumberTraits::log(ConstRef value) const
{
    RPY_CHECK(value.is_valid());
    RPY_CHECK_EQ(value.type(), *p_type);

    Value result(p_type);
    unsafe_exp(result.data(), value.data());

    return result;
}
Value NumberTraits::from_rational(int64_t numerator, int64_t denominator) const
{
    RPY_CHECK_NE(denominator, 0, std::domain_error);

    Value result(p_type);
    unsafe_from_rational(result.data(), numerator, denominator);

    return result;
}
