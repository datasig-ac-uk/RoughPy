//
// Created by sam on 27/11/24.
//

#include "polynomial_comparison.h"


#include "roughpy/core/debug_assertion.h"

#include "polynomial.h"



using namespace rpy;
using namespace rpy::generics;

bool PolynomialComparison::has_comparison(ComparisonType comp) const noexcept
{
    if (comp == ComparisonType::Equal) { return true; }
    return false;
}

bool PolynomialComparison::unsafe_compare_equal(const void* lhs,
    const void* rhs) const noexcept
{
    RPY_DBG_ASSERT_NE(lhs, nullptr);
    RPY_DBG_ASSERT_NE(rhs, nullptr);

    auto* lhs_ptr = static_cast<const Polynomial*>(lhs);
    auto* rhs_ptr = static_cast<const Polynomial*>(rhs);

    return poly_cmp_equal(*lhs_ptr, *rhs_ptr);
}

bool PolynomialComparison::unsafe_compare_less(const void* lhs,
    const void* rhs) const noexcept
{
    return false;
}

bool PolynomialComparison::unsafe_compare_less_equal(const void* lhs,
    const void* rhs) const noexcept
{
    return false;
}

bool PolynomialComparison::unsafe_compare_greater(const void* lhs,
    const void* rhs) const noexcept
{
    return false;
}

bool PolynomialComparison::unsafe_compare_greater_equal(const void* lhs,
    const void* rhs) const noexcept
{
    return false;
}
