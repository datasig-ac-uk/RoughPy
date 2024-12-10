//
// Created by sammorley on 25/11/24.
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

class TestRationalType : public ::testing::Test
{
protected:
    void SetUp() override { rational_type = MultiPrecisionTypes::get().rational_type; }

public:
    const ComparisonTrait* comp_trait() const noexcept
    {
        return trait_cast<const ComparisonTrait>(
                rational_type->get_builtin_trait(BuiltinTraitID::Comparison)
        );
    }

    const ArithmeticTrait* arith_trait() const noexcept
    {
        return trait_cast<const ArithmeticTrait>(
                rational_type->get_builtin_trait(BuiltinTraitID::Arithmetic)
        );
    }

    const NumberTrait* num_trait() const noexcept
    {
        return trait_cast<const NumberTrait>(
                rational_type->get_builtin_trait(BuiltinTraitID::Number)
        );
    }

    TypePtr rational_type;
};

}// namespace

TEST_F(TestRationalType, TestID)
{
    EXPECT_EQ(rational_type->id(), "apr");
    EXPECT_EQ(rational_type->name(), "rational");
}

TEST_F(TestRationalType, TestRefCounting)
{
    EXPECT_EQ(rational_type->ref_count(), 1);

    {
        RPY_MAYBE_UNUSED TypePtr new_ref(rational_type);
        EXPECT_EQ(rational_type->ref_count(), 1);
    }

    EXPECT_EQ(rational_type->ref_count(), 1);
}

TEST_F(TestRationalType, TestBasicProperties)
{
    EXPECT_GE(rational_type->object_size(), sizeof(void*));

    EXPECT_FALSE(concepts::is_standard_layout(*rational_type));
    EXPECT_FALSE(concepts::is_trivially_copyable(*rational_type));
    EXPECT_FALSE(concepts::is_trivially_constructible(*rational_type));
    EXPECT_FALSE(concepts::is_trivially_default_constructible(*rational_type));
    EXPECT_FALSE(concepts::is_trivially_copy_constructible(*rational_type));
    EXPECT_FALSE(concepts::is_trivially_copy_assignable(*rational_type));
    EXPECT_FALSE(concepts::is_trivially_destructible(*rational_type));
    EXPECT_FALSE(concepts::is_polymorphic(*rational_type));
    EXPECT_TRUE(concepts::is_signed(*rational_type));
    EXPECT_FALSE(concepts::is_unsigned(*rational_type));
    EXPECT_FALSE(concepts::is_integral(*rational_type));
    EXPECT_FALSE(concepts::is_arithmetic(*rational_type));
}

TEST_F(TestRationalType, TestDisplayAndParseFromString)
{
    Value value(rational_type, string_view("253/17"));
    std::stringstream ss;
    rational_type->display(ss, value.data());
    EXPECT_EQ(ss.str(), "253/17");
}

/******************************************************************************
 *                                Comparison                                  *
 ******************************************************************************/

TEST_F(TestRationalType, TestHasEqualOperator)
{
    auto* comp = comp_trait();
    ASSERT_NE(comp, nullptr);

    EXPECT_TRUE(comp->has_comparison(ComparisonType::Equal));
}

TEST_F(TestRationalType, TestHasLessThanOperator)
{
    auto* comp = comp_trait();
    ASSERT_NE(comp, nullptr);
    EXPECT_TRUE(comp->has_comparison(ComparisonType::Less));
}

TEST_F(TestRationalType, TestHasGreaterThanOperator)
{
    auto* comp = comp_trait();
    ASSERT_NE(comp, nullptr);
    EXPECT_TRUE(comp->has_comparison(ComparisonType::Greater));
}

TEST_F(TestRationalType, TestHasLessThanOrEqualOperator)
{
    auto* comp = comp_trait();
    ASSERT_NE(comp, nullptr);
    EXPECT_TRUE(comp->has_comparison(ComparisonType::LessEqual));
}

TEST_F(TestRationalType, TestHasGreaterThanOrEqualOperator)
{
    auto* comp = comp_trait();
    ASSERT_NE(comp, nullptr);
    EXPECT_TRUE(comp->has_comparison(ComparisonType::GreaterEqual));
}

/******************************************************************************
 *                                Arithmetic                                  *
 ******************************************************************************/

TEST_F(TestRationalType, TestAddOperator)
{
    auto* arith = arith_trait();
    ASSERT_NE(arith, nullptr);

    EXPECT_TRUE(arith->has_operation(ArithmeticOperation::Add));
}

TEST_F(TestRationalType, TestSubtractOperator)
{
    auto* arith = arith_trait();
    ASSERT_NE(arith, nullptr);

    EXPECT_TRUE(arith->has_operation(ArithmeticOperation::Sub));
}

TEST_F(TestRationalType, TestMultiplyOperator)
{
    auto* arith = arith_trait();
    ASSERT_NE(arith, nullptr);

    EXPECT_TRUE(arith->has_operation(ArithmeticOperation::Mul));
}

TEST_F(TestRationalType, TestDivideOperator)
{
    auto* arith = arith_trait();
    ASSERT_NE(arith, nullptr);

    EXPECT_TRUE(arith->has_operation(ArithmeticOperation::Div));
}

/******************************************************************************
 *                                Number-like                                 *
 ******************************************************************************/

TEST_F(TestRationalType, TestRealAndImaginary)
{
    const auto* num = num_trait();
    ASSERT_NE(num, nullptr);

    EXPECT_TRUE(num->has_function(NumberFunction::Real));
    EXPECT_TRUE(num->has_function(NumberFunction::Imaginary));

}

TEST_F(TestRationalType, TestAbsFunction)
{
    const auto* num = num_trait();
    ASSERT_NE(num, nullptr);
    EXPECT_TRUE(num->has_function(NumberFunction::Abs));
}

TEST_F(TestRationalType, TestSqrtExpLogFunction)
{
    // Rational numbers do not have sqrt, exp, or log.
    const auto* num = num_trait();
    ASSERT_NE(num, nullptr);

    EXPECT_FALSE(num->has_function(NumberFunction::Sqrt));
    EXPECT_FALSE(num->has_function(NumberFunction::Exp));
    EXPECT_FALSE(num->has_function(NumberFunction::Log));
}

TEST_F(TestRationalType, TestPowFunction)
{
    const auto* num = num_trait();
    ASSERT_NE(num, nullptr);
    EXPECT_TRUE(num->has_function(NumberFunction::Pow));
}


TEST_F(TestRationalType, TestFromRationalFunction)
{
    const auto* num = num_trait();
    ASSERT_NE(num, nullptr);
    EXPECT_TRUE(num->has_function(NumberFunction::FromRational));

}