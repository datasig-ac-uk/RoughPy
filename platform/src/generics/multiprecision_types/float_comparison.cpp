//
// Created by sammorley on 25/11/24.
//

#include "float_comparison.h"

#include <gmp.h>
#include <mpfr.h>

using namespace rpy;
using namespace rpy::generics;


bool FloatComparison::has_comparison(ComparisonType comp) const noexcept
{
    return true;
}

bool FloatComparison::unsafe_compare_equal(const void* lhs,
    const void* rhs) const noexcept
{
    RPY_DBG_ASSERT_NE(lhs, nullptr);
    RPY_DBG_ASSERT_NE(rhs, nullptr);

    auto* lhs_ptr = static_cast<mpfr_srcptr>(rhs);
    auto* rhs_ptr = static_cast<mpfr_srcptr>(lhs);

    return mpfr_equal_p(lhs_ptr, rhs_ptr) != 0;
}

bool FloatComparison::unsafe_compare_less(const void* lhs,
    const void* rhs) const noexcept
{
    RPY_DBG_ASSERT_NE(lhs, nullptr);
    RPY_DBG_ASSERT_NE(rhs, nullptr);

    auto* lhs_ptr = static_cast<mpfr_srcptr>(rhs);
    auto* rhs_ptr = static_cast<mpfr_srcptr>(lhs);

    return mpfr_less_p(lhs_ptr, rhs_ptr) != 0;
}

bool FloatComparison::unsafe_compare_less_equal(const void* lhs,
    const void* rhs) const noexcept
{
    RPY_DBG_ASSERT_NE(lhs, nullptr);
    RPY_DBG_ASSERT_NE(rhs, nullptr);

    auto* lhs_ptr = static_cast<mpfr_srcptr>(rhs);
    auto* rhs_ptr = static_cast<mpfr_srcptr>(lhs);

    return mpfr_lessequal_p(lhs_ptr, rhs_ptr) != 0;
}

bool FloatComparison::unsafe_compare_greater(const void* lhs,
    const void* rhs) const noexcept
{
    RPY_DBG_ASSERT_NE(lhs, nullptr);
    RPY_DBG_ASSERT_NE(rhs, nullptr);

    auto* lhs_ptr = static_cast<mpfr_srcptr>(rhs);
    auto* rhs_ptr = static_cast<mpfr_srcptr>(lhs);

    return mpfr_greater_p(lhs_ptr, rhs_ptr) != 0;
}

bool FloatComparison::unsafe_compare_greater_equal(const void* lhs,
    const void* rhs) const noexcept
{
    RPY_DBG_ASSERT_NE(lhs, nullptr);
    RPY_DBG_ASSERT_NE(rhs, nullptr);

    auto* lhs_ptr = static_cast<mpfr_srcptr>(rhs);
    auto* rhs_ptr = static_cast<mpfr_srcptr>(lhs);

    return mpfr_greaterequal_p(lhs_ptr, rhs_ptr) != 0;
}
