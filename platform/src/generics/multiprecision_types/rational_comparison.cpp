//
// Created by sammorley on 25/11/24.
//

#include "rational_comparison.h"

#include <gmp.h>


using namespace rpy;
using namespace rpy::generics;

bool RationalComparison::has_comparison(ComparisonType comp) const noexcept
{
    return true;
}
bool RationalComparison::unsafe_compare_equal(
        const void* lhs,
        const void* rhs
) const noexcept
{
    RPY_DBG_ASSERT_NE(lhs, nullptr);
    RPY_DBG_ASSERT_NE(rhs, nullptr);

    auto* lhs_ptr = static_cast<mpq_srcptr>(lhs);
    auto* rhs_ptr = static_cast<mpq_srcptr>(rhs);

    return mpq_equal(lhs_ptr, rhs_ptr) != 0;
}
bool RationalComparison::unsafe_compare_less(
        const void* lhs,
        const void* rhs
) const noexcept
{
    RPY_DBG_ASSERT_NE(lhs, nullptr);
    RPY_DBG_ASSERT_NE(rhs, nullptr);

    auto* lhs_ptr = static_cast<mpq_srcptr>(lhs);
    auto* rhs_ptr = static_cast<mpq_srcptr>(rhs);

    return mpq_cmp(lhs_ptr, rhs_ptr) < 0;
}
bool RationalComparison::unsafe_compare_less_equal(
        const void* lhs,
        const void* rhs
) const noexcept
{
    RPY_DBG_ASSERT_NE(lhs, nullptr);
    RPY_DBG_ASSERT_NE(rhs, nullptr);

    auto* lhs_ptr = static_cast<mpq_srcptr>(lhs);
    auto* rhs_ptr = static_cast<mpq_srcptr>(rhs);

    return mpq_cmp(lhs_ptr, rhs_ptr) <= 0;
}
bool RationalComparison::unsafe_compare_greater(
        const void* lhs,
        const void* rhs
) const noexcept
{
    RPY_DBG_ASSERT_NE(lhs, nullptr);
    RPY_DBG_ASSERT_NE(rhs, nullptr);

    auto* lhs_ptr = static_cast<mpq_srcptr>(lhs);
    auto* rhs_ptr = static_cast<mpq_srcptr>(rhs);

    return mpq_cmp(lhs_ptr, rhs_ptr) > 0;
}
bool RationalComparison::unsafe_compare_greater_equal(
        const void* lhs,
        const void* rhs
) const noexcept
{
    RPY_DBG_ASSERT_NE(lhs, nullptr);
    RPY_DBG_ASSERT_NE(rhs, nullptr);

    auto* lhs_ptr = static_cast<mpq_srcptr>(lhs);
    auto* rhs_ptr = static_cast<mpq_srcptr>(rhs);

    return mpq_cmp(lhs_ptr, rhs_ptr) >= 0;
}
