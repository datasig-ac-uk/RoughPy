//
// Created by sammorley on 25/11/24.
//

#include "float_number.h"

#include <mpfr.h>

using namespace rpy;
using namespace rpy::generics;

bool FloatNumber::has_function(NumberFunction fn_id) const noexcept
{
    return true;
}

void FloatNumber::unsafe_real(void* dst, const void* src) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_DBG_ASSERT_NE(src, nullptr);

    auto* src_ptr = static_cast<mpfr_srcptr>(src);
    auto* dst_ptr = static_cast<mpfr_ptr>(dst);

    mpfr_set(dst_ptr, src_ptr, MPFR_RNDN);
}

void FloatNumber::unsafe_imaginary(void* dst, const void* src) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    ignore_unused(src);

    auto* dst_ptr = static_cast<mpfr_ptr>(dst);

    mpfr_set_zero(dst_ptr, 1);
}

void FloatNumber::unsafe_abs(void* dst, const void* src) const noexcept
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_DBG_ASSERT_NE(src, nullptr);

    auto* src_ptr = static_cast<mpfr_srcptr>(src);
    auto* dst_ptr = static_cast<mpfr_ptr>(dst);

    mpfr_abs(dst_ptr, src_ptr, MPFR_RNDN);
}

void FloatNumber::unsafe_sqrt(void* dst, const void* src) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_DBG_ASSERT_NE(src, nullptr);

    auto* src_ptr = static_cast<mpfr_srcptr>(src);
    auto* dst_ptr = static_cast<mpfr_ptr>(dst);

    mpfr_sqrt(dst_ptr, src_ptr, MPFR_RNDN);
}

void FloatNumber::unsafe_pow(void* dst,
    const void* base,
    exponent_t exponent) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_DBG_ASSERT_NE(base, nullptr);

    auto* base_ptr = static_cast<mpfr_srcptr>(base);
    auto* dst_ptr = static_cast<mpfr_ptr>(dst);

    mpfr_pow_si(dst_ptr, base_ptr, exponent, MPFR_RNDN);
}

void FloatNumber::unsafe_exp(void* dst, const void* src) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_DBG_ASSERT_NE(src, nullptr);

    auto* src_ptr = static_cast<mpfr_srcptr>(src);
    auto* dst_ptr = static_cast<mpfr_ptr>(dst);

    mpfr_exp(dst_ptr, src_ptr, MPFR_RNDN);
}

void FloatNumber::unsafe_log(void* dst, const void* src) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_DBG_ASSERT_NE(src, nullptr);

    auto* src_ptr = static_cast<mpfr_srcptr>(src);
    auto* dst_ptr = static_cast<mpfr_ptr>(dst);

    mpfr_log(dst_ptr, src_ptr, MPFR_RNDN);
}

void FloatNumber::unsafe_from_rational(void* dst,
    int64_t numerator,
    int64_t denominator) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_CHECK_NE(denominator, 0);

    if (numerator < 0) {
        numerator = -numerator;
        denominator = -denominator;
    }

    auto* dst_ptr = static_cast<mpfr_ptr>(dst);

    mpfr_set_si(dst_ptr, numerator, MPFR_RNDN);
    mpfr_div_si(dst_ptr, dst_ptr, denominator, MPFR_RNDN);
}
