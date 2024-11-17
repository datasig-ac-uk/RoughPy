//
// Created by sam on 14/11/24.
//

#include "roughpy/generics/arithmetic_trait.h"

#include "roughpy/core/debug_assertion.h"
#include "roughpy/core/check.h"

#include "roughpy/generics/type.h"
#include "roughpy/generics/values.h"

using namespace rpy;
using namespace rpy::generics;
ArithmeticTrait::~ArithmeticTrait() = default;
bool ArithmeticTrait::has_operation(ArithmeticOperation op) const noexcept
{
    return false;
}
void ArithmeticTrait::add_inplace(Ref lhs, ConstRef rhs) const
{
    RPY_CHECK(has_operation(ArithmeticOperation::Add));
    RPY_CHECK(rhs.is_valid() && lhs.is_valid());
    RPY_CHECK_EQ(*p_type, lhs.type());

    if (*p_type == rhs.type()) {
        unsafe_add_inplace(lhs.data(), rhs.data());
    } else {
        Value tmp(p_type);
        tmp = rhs;
        unsafe_add_inplace(lhs.data(), tmp.data());
    }
}
void ArithmeticTrait::sub_inplace(Ref lhs, ConstRef rhs) const
{
    RPY_CHECK(has_operation(ArithmeticOperation::Sub));
    RPY_CHECK(rhs.is_valid() && lhs.is_valid());
    RPY_CHECK_EQ(*p_type, lhs.type());

    if (*p_type == rhs.type()) {
        unsafe_sub_inplace(lhs.data(), rhs.data());
    } else {
        Value tmp(p_type);
        tmp = rhs;
        unsafe_sub_inplace(lhs.data(), tmp.data());
    }
}
void ArithmeticTrait::mul_inplace(Ref lhs, ConstRef rhs) const
{
    RPY_CHECK(has_operation(ArithmeticOperation::Mul));
    RPY_CHECK(rhs.is_valid() && lhs.is_valid());
    RPY_CHECK_EQ(*p_type, lhs.type());

    if (*p_type == rhs.type()) {
        unsafe_mul_inplace(lhs.data(), rhs.data());
    } else {
        Value tmp(p_type);
        tmp = rhs;
        unsafe_mul_inplace(lhs.data(), tmp.data());
    }

}
void ArithmeticTrait::div_inplace(Ref lhs, ConstRef rhs) const
{
    RPY_CHECK(has_operation(ArithmeticOperation::Div));
    RPY_CHECK(rhs.is_valid());
    RPY_CHECK(!rhs.fast_is_zero(), "division by zero", std::domain_error);

    RPY_CHECK_EQ(*p_type, rhs.type());

    if (rhs.type() == *p_rational_type) {
        unsafe_div_inplace(lhs.data(), rhs.data());
    } else {
        Value tmp(p_rational_type);
        tmp = rhs;
        unsafe_div_inplace(lhs.data(), tmp.data());
    }

}
Value ArithmeticTrait::add(ConstRef lhs, ConstRef rhs) const
{
    RPY_CHECK(has_operation(ArithmeticOperation::Add));
    RPY_CHECK(rhs.is_valid() && lhs.is_valid());
    Value result(lhs);
    unsafe_add_inplace(result.data(), rhs.data());
    return result;
}
Value ArithmeticTrait::sub(ConstRef lhs, ConstRef rhs) const
{
    RPY_CHECK(has_operation(ArithmeticOperation::Sub));
    RPY_CHECK(rhs.is_valid() && lhs.is_valid());
    Value result(lhs);
    unsafe_sub_inplace(result.data(), rhs.data());
    return result;
}
Value ArithmeticTrait::mul(ConstRef lhs, ConstRef rhs) const
{
    RPY_CHECK(has_operation(ArithmeticOperation::Mul));
    RPY_CHECK(rhs.is_valid() && lhs.is_valid());
    Value result(lhs);
    unsafe_mul_inplace(result.data(), rhs.data());
    return result;
}
Value ArithmeticTrait::div(ConstRef lhs, ConstRef rhs) const
{
    RPY_CHECK(has_operation(ArithmeticOperation::Div));
    RPY_CHECK(rhs.is_valid() && lhs.is_valid());
    Value result(lhs);

    unsafe_div_inplace(result.data(), rhs.data());
    return result;
}
