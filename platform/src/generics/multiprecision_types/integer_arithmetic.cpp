//
// Created by sammorley on 25/11/24.
//

#include "integer_arithmetic.h"

#include <gmp.h>

using namespace rpy;
using namespace rpy::generics;

bool IntegerArithmetic::has_operation(ArithmeticOperation op) const noexcept
{
    if (op == ArithmeticOperation::Div) {
        return false;
    }
    return true;
}
void IntegerArithmetic::unsafe_add_inplace(
        void* lhs,
        const void* rhs
) const noexcept
{
    RPY_DBG_ASSERT_NE(lhs, nullptr);
    RPY_DBG_ASSERT_NE(rhs, nullptr);

    auto* src_ptr = static_cast<mpz_srcptr>(rhs);
    auto* dst_ptr = static_cast<mpz_ptr>(lhs);

    mpz_add(dst_ptr, dst_ptr, src_ptr);
}
void IntegerArithmetic::unsafe_sub_inplace(void* lhs, const void* rhs) const
{
    RPY_DBG_ASSERT_NE(lhs, nullptr);
    RPY_DBG_ASSERT_NE(rhs, nullptr);

    auto* src_ptr = static_cast<mpz_srcptr>(rhs);
    auto* dst_ptr = static_cast<mpz_ptr>(lhs);

    mpz_sub(dst_ptr, dst_ptr, src_ptr);
}
void IntegerArithmetic::unsafe_mul_inplace(void* lhs, const void* rhs) const
{
    RPY_DBG_ASSERT_NE(lhs, nullptr);
    RPY_DBG_ASSERT_NE(rhs, nullptr);

    auto* src_ptr = static_cast<mpz_srcptr>(rhs);
    auto* dst_ptr = static_cast<mpz_ptr>(lhs);

    mpz_mul(dst_ptr, dst_ptr, src_ptr);
}
void IntegerArithmetic::unsafe_div_inplace(void* lhs, const void* rhs) const {}
