//
// Created by sammorley on 25/11/24.
//

#include "rational_arithmetic.h"

#include <gmp.h>

using namespace rpy;
using namespace rpy::generics;


bool RationalArithmetic::has_operation(ArithmeticOperation op) const noexcept
{
    return true;
}
void RationalArithmetic::unsafe_add_inplace(
        void* lhs,
        const void* rhs
) const noexcept
{
    RPY_DBG_ASSERT_NE(lhs, nullptr);
    RPY_DBG_ASSERT_NE(rhs, nullptr);

    auto* src_ptr = static_cast<mpq_srcptr>(lhs);
    auto* dst_ptr = static_cast<mpq_ptr>(lhs);

    mpq_add(dst_ptr, dst_ptr, src_ptr);
}
void RationalArithmetic::unsafe_sub_inplace(void* lhs, const void* rhs) const
{
    RPY_DBG_ASSERT_NE(lhs, nullptr);
    RPY_DBG_ASSERT_NE(rhs, nullptr);

    auto* src_ptr = static_cast<mpq_srcptr>(rhs);
    auto* dst_ptr = static_cast<mpq_ptr>(lhs);

    mpq_sub(dst_ptr, dst_ptr, src_ptr);
}
void RationalArithmetic::unsafe_mul_inplace(void* lhs, const void* rhs) const
{
    RPY_DBG_ASSERT_NE(lhs, nullptr);
    RPY_DBG_ASSERT_NE(rhs, nullptr);

    auto* src_ptr = static_cast<mpq_srcptr>(rhs);
    auto* dst_ptr = static_cast<mpq_ptr>(lhs);

    mpq_mul(dst_ptr, dst_ptr, src_ptr);
}
void RationalArithmetic::unsafe_div_inplace(void* lhs, const void* rhs) const
{
    RPY_DBG_ASSERT_NE(lhs, nullptr);
    RPY_DBG_ASSERT_NE(rhs, nullptr);

    auto* src_ptr = static_cast<mpq_srcptr>(rhs);
    auto* dst_ptr = static_cast<mpq_ptr>(lhs);

    mpq_div(dst_ptr, dst_ptr, src_ptr);
}