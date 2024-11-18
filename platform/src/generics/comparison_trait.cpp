//
// Created by sam on 14/11/24.
//

#include "roughpy/generics/comparison_trait.h"

#include "roughpy/generics/type.h"
#include "roughpy/generics/values.h"

using namespace rpy;
using namespace rpy::generics;

ComparisonTrait::~ComparisonTrait() = default;
bool ComparisonTrait::compare_equal(ConstRef lhs, ConstRef rhs) const
{
    RPY_CHECK(has_comparison(ComparisonType::Equal));

    if (!lhs.is_valid() && !rhs.is_valid()) {
        return true;
    }

    if (lhs.is_valid() || rhs.is_valid()) {
        return false;
    }

    RPY_CHECK_EQ(lhs.type(), rhs.type());

    return unsafe_compare_equal(lhs.data(), rhs.data());
}
bool ComparisonTrait::compare_less(ConstRef lhs, ConstRef rhs) const
{
    RPY_CHECK(has_comparison(ComparisonType::Less));

    RPY_CHECK(!lhs.fast_is_zero());
    RPY_CHECK(!rhs.fast_is_zero());

    RPY_CHECK_EQ(lhs.type(), rhs.type());

    return unsafe_compare_less(lhs.data(), rhs.data());
}
bool ComparisonTrait::compare_less_equal(ConstRef lhs, ConstRef rhs) const
{
    RPY_CHECK(has_comparison(ComparisonType::LessEqual));
    RPY_CHECK(!lhs.fast_is_zero());
    RPY_CHECK(!rhs.fast_is_zero());

    RPY_CHECK_EQ(lhs.type(), rhs.type());

    return unsafe_compare_less_equal(lhs.data(), rhs.data());
}
bool ComparisonTrait::compare_less_greater(ConstRef lhs, ConstRef rhs) const
{
    RPY_CHECK(has_comparison(ComparisonType::Greater));
    RPY_CHECK(!lhs.fast_is_zero());
    RPY_CHECK(!rhs.fast_is_zero());

    RPY_CHECK_EQ(lhs.type(), rhs.type());

    return unsafe_compare_greater(lhs.data(), rhs.data());
}
bool ComparisonTrait::compare_less_greater_equal(ConstRef lhs, ConstRef rhs)
        const
{
    RPY_CHECK(has_comparison(ComparisonType::GreaterEqual));
    RPY_CHECK(!lhs.fast_is_zero());
    RPY_CHECK(!rhs.fast_is_zero());

    RPY_CHECK_EQ(lhs.type(), rhs.type());

    return unsafe_compare_greater_equal(lhs.data(), rhs.data());
}

