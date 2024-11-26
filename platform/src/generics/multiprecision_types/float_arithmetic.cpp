//
// Created by sammorley on 25/11/24.
//

#include "float_arithmetic.h"

#include <gmp.h>
#include <mpfr.h>


using namespace rpy;
using namespace rpy::generics;


bool FloatArithmetic::has_operation(ArithmeticOperation op) const noexcept
{
    return true;
}

void FloatArithmetic::unsafe_add_inplace(void* lhs,
    const void* rhs) const noexcept
{
    RPY_DBG_ASSERT_NE(lhs, nullptr);
    RPY_DBG_ASSERT_NE(rhs, nullptr);

    auto* src_ptr = static_cast<mpfr_srcptr>(rhs);
    auto* dst_ptr = static_cast<mpfr_ptr>(lhs);

    mpfr_add(dst_ptr, dst_ptr, src_ptr, MPFR_RNDN);
}

void FloatArithmetic::unsafe_sub_inplace(void* lhs, const void* rhs) const
{
    RPY_DBG_ASSERT_NE(lhs, nullptr);
    RPY_DBG_ASSERT_NE(rhs, nullptr);

    auto* src_ptr = static_cast<mpfr_srcptr>(rhs);
    auto* dst_ptr = static_cast<mpfr_ptr>(lhs);

    mpfr_sub(dst_ptr, dst_ptr, src_ptr, MPFR_RNDN);
}

void FloatArithmetic::unsafe_mul_inplace(void* lhs, const void* rhs) const
{
    RPY_DBG_ASSERT_NE(lhs, nullptr);
    RPY_DBG_ASSERT_NE(rhs, nullptr);

    auto* src_ptr = static_cast<mpfr_srcptr>(rhs);
    auto* dst_ptr = static_cast<mpfr_ptr>(lhs);

    mpfr_mul(dst_ptr, dst_ptr, src_ptr, MPFR_RNDN);
}

void FloatArithmetic::unsafe_div_inplace(void* lhs, const void* rhs) const
{
    RPY_DBG_ASSERT_NE(lhs, nullptr);
    RPY_DBG_ASSERT_NE(rhs, nullptr);

    auto* src_ptr = static_cast<mpfr_srcptr>(rhs);
    auto* dst_ptr = static_cast<mpfr_ptr>(lhs);

    mpfr_div(dst_ptr, dst_ptr, src_ptr, MPFR_RNDN);
}
