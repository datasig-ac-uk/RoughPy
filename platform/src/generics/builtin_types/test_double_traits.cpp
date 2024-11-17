//
// Created by sam on 17/11/24.
//



#include <gtest/gtest.h>

#include <roughpy/generics/comparison_trait.h>
#include <roughpy/generics/arithmetic_trait.h>

using namespace rpy;
using namespace rpy::generics;

TEST(TestDouble, TestHasEqualOperator)
{
    const auto double_type = get_type<double>();
    ComparisonTraitImpl<double> traits(double_type.get());
    EXPECT_TRUE(traits.has_comparison(ComparisonType::Equal));
}

TEST(TestDouble, TestHasLessOperator)
{
    const auto double_type = get_type<double>();
    ComparisonTraitImpl<double> traits(double_type.get());
    EXPECT_TRUE(traits.has_comparison(ComparisonType::Less));
}

TEST(TestDouble, TestHasLessEqualOperator)
{
    const auto double_type = get_type<double>();
    ComparisonTraitImpl<double> traits(double_type.get());
    EXPECT_TRUE(traits.has_comparison(ComparisonType::LessEqual));
}

TEST(TestDouble, TestHasGreaterOperator)
{
    const auto double_type = get_type<double>();
    ComparisonTraitImpl<double> traits(double_type.get());
    EXPECT_TRUE(traits.has_comparison(ComparisonType::Greater));
}

TEST(TestDouble, TestHasGreaterEqualOperator)
{
    const auto double_type = get_type<double>();
    ComparisonTraitImpl<double> traits(double_type.get());
    EXPECT_TRUE(traits.has_comparison(ComparisonType::GreaterEqual));
}
