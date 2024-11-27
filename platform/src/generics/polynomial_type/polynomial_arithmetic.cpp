//
// Created by sam on 27/11/24.
//

#include "polynomial_arithmetic.h"

#include "roughpy/core/check.h"
#include "roughpy/core/debug_assertion.h"

#include "polynomial.h"

bool rpy::generics::PolynomialArithmetic::
has_operation(ArithmeticOperation op) const noexcept { return true; }

void rpy::generics::PolynomialArithmetic::unsafe_add_inplace(void* lhs,
    const void* rhs) const noexcept
{
    RPY_DBG_ASSERT_NE(lhs, nullptr);
    RPY_DBG_ASSERT_NE(rhs, nullptr);

    auto* src_ptr = static_cast<const Polynomial*>(rhs);
    auto* dst_ptr = static_cast<Polynomial*>(lhs);

    poly_add_inplace(*dst_ptr, *src_ptr);
}

void rpy::generics::PolynomialArithmetic::unsafe_sub_inplace(void* lhs,
    const void* rhs) const
{
    RPY_DBG_ASSERT_NE(lhs, nullptr);
    RPY_DBG_ASSERT_NE(rhs, nullptr);

    auto* src_ptr = static_cast<const Polynomial*>(rhs);
    auto* dst_ptr = static_cast<Polynomial*>(lhs);

    poly_sub_inplace(*dst_ptr, *src_ptr);
}

void rpy::generics::PolynomialArithmetic::unsafe_mul_inplace(void* lhs,
    const void* rhs) const
{
    RPY_DBG_ASSERT_NE(lhs, nullptr);
    RPY_DBG_ASSERT_NE(rhs, nullptr);

    auto* src_ptr = static_cast<const Polynomial*>(rhs);
    auto* dst_ptr = static_cast<Polynomial*>(lhs);

    poly_mul_inplace(*dst_ptr, *src_ptr);
}

void rpy::generics::PolynomialArithmetic::unsafe_div_inplace(void* lhs,
    const void* rhs) const
{
    RPY_DBG_ASSERT_NE(lhs, nullptr);
    RPY_DBG_ASSERT_NE(rhs, nullptr);

    const auto* src_ptr = static_cast<mpq_srcptr>(rhs);
    auto* dst_ptr = static_cast<Polynomial*>(lhs);

    poly_div_inplace(*dst_ptr, src_ptr);
}