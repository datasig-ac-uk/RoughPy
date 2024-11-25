//
// Created by sammorley on 25/11/24.
//

#include "rational_number.h"

#include <gmp.h>


#include "roughpy/core/debug_assertion.h"
#include "roughpy/core/macros.h"
#include "roughpy/core/types.h"

using namespace rpy;
using namespace rpy::generics;

bool RationalNumber::has_function(NumberFunction fn_id) const noexcept
{
    switch (fn_id) {
        case NumberFunction::Abs: RPY_FALLTHROUGH;
        case NumberFunction::Pow: RPY_FALLTHROUGH;
        case NumberFunction::FromRational: RPY_FALLTHROUGH;
        case NumberFunction::Real: RPY_FALLTHROUGH;
        case NumberFunction::Imaginary: return true;
        case NumberFunction::Sqrt: RPY_FALLTHROUGH;
        case NumberFunction::Exp: RPY_FALLTHROUGH;
        case NumberFunction::Log: return false;
    }
    RPY_UNREACHABLE_RETURN(false);
}
void RationalNumber::unsafe_real(void* dst, const void* src) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_DBG_ASSERT_NE(src, nullptr);

    auto* src_ptr = static_cast<mpq_srcptr>(src);
    auto* dst_ptr = static_cast<mpq_ptr>(dst);

    mpq_set(dst_ptr, src_ptr);
}
void RationalNumber::unsafe_imaginary(void* dst, const void* src) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    ignore_unused(src);

    auto* dst_ptr = static_cast<mpq_ptr>(dst);

    mpq_set_si(dst_ptr, 0, 1);
}
void RationalNumber::unsafe_abs(void* dst, const void* src) const noexcept
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_DBG_ASSERT_NE(src, nullptr);

    auto* src_ptr = static_cast<mpq_srcptr>(src);
    auto* dst_ptr = static_cast<mpq_ptr>(dst);

    mpq_abs(dst_ptr, src_ptr);
}
void RationalNumber::unsafe_pow(
        void* dst,
        const void* base,
        exponent_t exponent
) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_DBG_ASSERT_NE(base, nullptr);

    auto* base_ptr = static_cast<mpq_srcptr>(base);
    auto* dst_ptr = static_cast<mpq_ptr>(dst);

    if (exponent == 0) {
        mpq_set_si(dst_ptr, 1, 1);
    } else if (exponent == 1) {
        mpq_set(dst_ptr, base_ptr);
    } else if (exponent == -1) {
        mpq_inv(dst_ptr, base_ptr);
    } else if (exponent > 0) {
        const auto expo = static_cast<uint64_t>(exponent);
        mpz_pow_ui(mpq_numref(dst_ptr), mpq_numref(base_ptr), expo);
        mpz_pow_ui(mpq_denref(dst_ptr), mpq_denref(base_ptr), expo);
    } else {
        const auto expo = static_cast<uint64_t>(exponent);
        mpz_pow_ui(mpq_numref(dst_ptr), mpq_denref(base_ptr), expo);
        mpz_pow_ui(mpq_denref(dst_ptr), mpq_numref(base_ptr), expo);
    }
}
void RationalNumber::unsafe_from_rational(
        void* dst,
        int64_t numerator,
        int64_t denominator
) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);

    // This should have been checked in the outer scope
    RPY_DBG_ASSERT_NE(numerator, 0);

    auto* dst_ptr = static_cast<mpq_ptr>(dst);

    if (denominator < 0) {
        numerator = -numerator;
        denominator = -denominator;
    }

    const auto denom = static_cast<uint64_t>(denominator);

    mpq_set_si(dst_ptr, numerator, denom);
}
