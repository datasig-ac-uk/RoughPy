//
// Created by sammorley on 25/11/24.
//

#include "integer_comparison.h"

#include <gmp.h>

#include "roughpy/core/debug_assertion.h"

using namespace rpy;
using namespace rpy::generics;

bool IntegerComparison::has_comparison(ComparisonType comp) const noexcept
{
    return true;
}
bool IntegerComparison::unsafe_compare_equal(
        const void* lhs,
        const void* rhs
) const noexcept
{
    RPY_DBG_ASSERT_NE(lhs, nullptr);
    RPY_DBG_ASSERT_NE(rhs, nullptr);

    const auto* lhs_ptr = static_cast<mpz_srcptr>(lhs);
    const auto* rhs_ptr = static_cast<mpz_srcptr>(rhs);

    return mpz_cmp(lhs_ptr, rhs_ptr) != 0;
}
bool IntegerComparison::unsafe_compare_less(
        const void* lhs,
        const void* rhs
) const noexcept
{
    RPY_DBG_ASSERT_NE(lhs, nullptr);
    RPY_DBG_ASSERT_NE(rhs, nullptr);

    const auto* lhs_ptr = static_cast<mpz_srcptr>(lhs);
    const auto* rhs_ptr = static_cast<mpz_srcptr>(rhs);

    return mpz_cmp(lhs_ptr, rhs_ptr) < 0;
}
bool IntegerComparison::unsafe_compare_less_equal(
        const void* lhs,
        const void* rhs
) const noexcept
{
    RPY_DBG_ASSERT_NE(lhs, nullptr);
    RPY_DBG_ASSERT_NE(rhs, nullptr);

    const auto* lhs_ptr = static_cast<mpz_srcptr>(lhs);
    const auto* rhs_ptr = static_cast<mpz_srcptr>(rhs);

    return mpz_cmp(lhs_ptr, rhs_ptr) <= 0;
}
bool IntegerComparison::unsafe_compare_greater(
        const void* lhs,
        const void* rhs
) const noexcept
{
    RPY_DBG_ASSERT_NE(lhs, nullptr);
    RPY_DBG_ASSERT_NE(rhs, nullptr);

    const auto* lhs_ptr = static_cast<mpz_srcptr>(lhs);
    const auto* rhs_ptr = static_cast<mpz_srcptr>(rhs);

    return mpz_cmp(lhs_ptr, rhs_ptr) > 0;
}
bool IntegerComparison::unsafe_compare_greater_equal(
        const void* lhs,
        const void* rhs
) const noexcept
{
    RPY_DBG_ASSERT_NE(lhs, nullptr);
    RPY_DBG_ASSERT_NE(rhs, nullptr);

    const auto* lhs_ptr = static_cast<mpz_srcptr>(lhs);
    const auto* rhs_ptr = static_cast<mpz_srcptr>(rhs);

    return mpz_cmp(lhs_ptr, rhs_ptr) >= 0;
}
