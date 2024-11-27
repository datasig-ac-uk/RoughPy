//
// Created by sam on 27/11/24.
//
#include <gtest/gtest.h>

#include "roughpy/generics/arithmetic_trait.h"
#include "roughpy/generics/builtin_trait.h"
#include "roughpy/generics/comparison_trait.h"
#include "roughpy/generics/number_trait.h"
#include "roughpy/generics/type.h"
#include "roughpy/generics/values.h"

using namespace rpy;
using namespace rpy::generics;

namespace {

class TestPolynomialType : public ::testing::Test
{
protected:
    void SetUp() override
    {
        polynomial_type = get_polynomial_type();
    }

public:
    const ComparisonTrait* comp_trait() const noexcept
    {
        return trait_cast<const ComparisonTrait>(
                polynomial_type->get_builtin_trait(BuiltinTraitID::Comparison)
        );
    }

    const ArithmeticTrait* arith_trait() const noexcept
    {
        return trait_cast<const ArithmeticTrait>(
                polynomial_type->get_builtin_trait(BuiltinTraitID::Arithmetic)
        );
    }

    const NumberTrait* num_trait() const noexcept
    {
        return trait_cast<const NumberTrait>(
                polynomial_type->get_builtin_trait(BuiltinTraitID::Number)
        );
    }

    TypePtr polynomial_type;
};

}// namespace

TEST_F(TestPolynomialType, TestID)
{
    EXPECT_EQ(polynomial_type->id(), "polynomial_id"); // Adjust ID accordingly
    EXPECT_EQ(polynomial_type->name(), "polynomial");
}

TEST_F(TestPolynomialType, TestRefCounting)
{
    EXPECT_EQ(polynomial_type->ref_count(), 1);

    {
        RPY_MAYBE_UNUSED TypePtr new_ref(polynomial_type);
        EXPECT_EQ(polynomial_type->ref_count(), 1);
    }

    EXPECT_EQ(polynomial_type->ref_count(), 1);
}

TEST_F(TestPolynomialType, TestBasicProperties)
{
    EXPECT_EQ(polynomial_type->object_size(), sizeof(void*)); // Adjust if necessary

    EXPECT_FALSE(concepts::is_standard_layout(*polynomial_type));
    EXPECT_FALSE(concepts::is_trivially_copyable(*polynomial_type));
    EXPECT_FALSE(concepts::is_trivially_constructible(*polynomial_type));
    EXPECT_FALSE(concepts::is_trivially_default_constructible(*polynomial_type));
    EXPECT_FALSE(concepts::is_trivially_copy_constructible(*polynomial_type));
    EXPECT_FALSE(concepts::is_trivially_copy_assignable(*polynomial_type));
    EXPECT_FALSE(concepts::is_trivially_destructible(*polynomial_type));
    EXPECT_FALSE(concepts::is_polymorphic(*polynomial_type));
    EXPECT_TRUE(concepts::is_signed(*polynomial_type));
    EXPECT_FALSE(concepts::is_unsigned(*polynomial_type));
    EXPECT_FALSE(concepts::is_integral(*polynomial_type));
    EXPECT_FALSE(concepts::is_arithmetic(*polynomial_type));
}

TEST_F(TestPolynomialType, TestDisplayAndParseFromString)
{
    Value value(polynomial_type, string_view("x^2+3*x+2")); // Example string
    std::stringstream ss;
    polynomial_type->display(ss, value.data());
    EXPECT_EQ(ss.str(), "x^2+3*x+2"); // Example string
}

/******************************************************************************
 *                                Comparison                                  *
 ******************************************************************************/

TEST_F(TestPolynomialType, TestHasEqualOperator)
{
    auto* comp = comp_trait();
    ASSERT_NE(comp, nullptr);

    EXPECT_TRUE(comp->has_comparison(ComparisonType::Equal));
}

TEST_F(TestPolynomialType, TestHasLessThanOperator)
{
    auto* comp = comp_trait();
    ASSERT_NE(comp, nullptr);
    EXPECT_FALSE(comp->has_comparison(ComparisonType::Less));
}

TEST_F(TestPolynomialType, TestHasGreaterThanOperator)
{
    auto* comp = comp_trait();
    ASSERT_NE(comp, nullptr);
    EXPECT_FALSE(comp->has_comparison(ComparisonType::Greater));
}

TEST_F(TestPolynomialType, TestHasLessThanOrEqualOperator)
{
    auto* comp = comp_trait();
    ASSERT_NE(comp, nullptr);
    EXPECT_FALSE(comp->has_comparison(ComparisonType::LessEqual));
}

TEST_F(TestPolynomialType, TestHasGreaterThanOrEqualOperator)
{
    auto* comp = comp_trait();
    ASSERT_NE(comp, nullptr);
    EXPECT_FALSE(comp->has_comparison(ComparisonType::GreaterEqual));
}

/******************************************************************************
 *                                Arithmetic                                  *
 ******************************************************************************/

TEST_F(TestPolynomialType, TestAddOperator)
{
    auto* arith = arith_trait();
    ASSERT_NE(arith, nullptr);

    EXPECT_TRUE(arith->has_operation(ArithmeticOperation::Add));
}

TEST_F(TestPolynomialType, TestSubtractOperator)
{
    auto* arith = arith_trait();
    ASSERT_NE(arith, nullptr);

    EXPECT_TRUE(arith->has_operation(ArithmeticOperation::Sub));
}

TEST_F(TestPolynomialType, TestMultiplyOperator)
{
    auto* arith = arith_trait();
    ASSERT_NE(arith, nullptr);

    EXPECT_TRUE(arith->has_operation(ArithmeticOperation::Mul));
}

TEST_F(TestPolynomialType, TestDivideOperator)
{
    auto* arith = arith_trait();
    ASSERT_NE(arith, nullptr);

    EXPECT_FALSE(arith->has_operation(ArithmeticOperation::Div)); // Depending on your polynomial type
}

/******************************************************************************
 *                                Number-like                                 *
 ******************************************************************************/

TEST_F(TestPolynomialType, TestRealAndImaginary)
{
    const auto* num = num_trait();
    ASSERT_NE(num, nullptr);

    EXPECT_FALSE(num->has_function(NumberFunction::Real));
    EXPECT_FALSE(num->has_function(NumberFunction::Imaginary));
}

TEST_F(TestPolynomialType, TestAbsFunction)
{
    const auto* num = num_trait();
    ASSERT_NE(num, nullptr);
    EXPECT_FALSE(num->has_function(NumberFunction::Abs));
}

TEST_F(TestPolynomialType, TestSqrtExpLogFunction)
{
    const auto* num = num_trait();
    ASSERT_NE(num, nullptr);

    EXPECT_FALSE(num->has_function(NumberFunction::Sqrt));
    EXPECT_FALSE(num->has_function(NumberFunction::Exp));
    EXPECT_FALSE(num->has_function(NumberFunction::Log));
}

TEST_F(TestPolynomialType, TestPowFunction)
{
    const auto* num = num_trait();
    ASSERT_NE(num, nullptr);
    EXPECT_FALSE(num->has_function(NumberFunction::Pow));
}

TEST_F(TestPolynomialType, TestFromRationalFunction)
{
    const auto* num = num_trait();
    ASSERT_NE(num, nullptr);
    EXPECT_FALSE(num->has_function(NumberFunction::FromRational));
}