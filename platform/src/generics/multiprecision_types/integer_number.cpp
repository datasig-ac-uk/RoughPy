//
// Created by sammorley on 25/11/24.
//

#include "integer_number.h"

#include <gmp.h>

#include "roughpy/core/debug_assertion.h"
#include "roughpy/core/macros.h"
#include "roughpy/core/types.h"

#include <boost/multiprecision/detail/assert.hpp>

using namespace rpy;
using namespace rpy::generics;

bool IntegerNumber::has_function(NumberFunction fn_id) const noexcept
{
    switch (fn_id) {
        case NumberFunction::Abs: RPY_FALLTHROUGH;
        case NumberFunction::Pow: RPY_FALLTHROUGH;
        case NumberFunction::Real: RPY_FALLTHROUGH;
        case NumberFunction::Imaginary: RPY_FALLTHROUGH;
        case NumberFunction::FromRational: return true;
        case NumberFunction::Sqrt: RPY_FALLTHROUGH;
        case NumberFunction::Exp: RPY_FALLTHROUGH;
        case NumberFunction::Log: return false;
    }
    RPY_UNREACHABLE_RETURN(false);
}
void IntegerNumber::unsafe_real(void* dst, const void* src) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_DBG_ASSERT_NE(src, nullptr);

    auto* src_ptr = static_cast<mpz_srcptr>(src);
    auto* dst_ptr = static_cast<mpz_ptr>(dst);

    mpz_set(dst_ptr, src_ptr);
}
void IntegerNumber::unsafe_imaginary(void* dst, const void* src) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    ignore_unused(src);

    auto* dst_ptr = static_cast<mpz_ptr>(dst);

    mpz_set_si(dst_ptr, 0);
}
void IntegerNumber::unsafe_abs(void* dst, const void* src) const noexcept
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_DBG_ASSERT_NE(src, nullptr);

    auto* src_ptr = static_cast<mpz_srcptr>(src);
    auto* dst_ptr = static_cast<mpz_ptr>(dst);

    mpz_abs(dst_ptr, src_ptr);
}

void IntegerNumber::unsafe_from_rational(
        void* dst,
        int64_t numerator,
        int64_t denominator
) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_CHECK_NE(denominator, 0);

    if (numerator < 0) {
        numerator = -numerator;
        denominator = -denominator;
    }

    RPY_CHECK_EQ(denominator, 1);

    auto* dst_ptr = static_cast<mpz_ptr>(dst);

    mpz_set_si(dst_ptr, numerator);
}
