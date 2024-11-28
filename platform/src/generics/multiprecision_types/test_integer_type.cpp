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

class TestIntegerType : public ::testing::Test
{
protected:

    void SetUp() override
    {
        integer_type = MultiPrecisionTypes::get().integer_type;

    }

public:
    const ComparisonTrait* comp_trait() const noexcept
    {
        return trait_cast<const ComparisonTrait>(
                integer_type->get_builtin_trait(BuiltinTraitID::Comparison)
        );
    }

    const ArithmeticTrait* arith_trait() const noexcept
    {
        return trait_cast<const ArithmeticTrait>(
                integer_type->get_builtin_trait(BuiltinTraitID::Arithmetic)
        );
    }

    const NumberTrait* num_trait() const noexcept
    {
        return trait_cast<const NumberTrait>(
                integer_type->get_builtin_trait(BuiltinTraitID::Number)
        );
    }


    TypePtr integer_type;

};


}



TEST_F(TestIntegerType, TestID)
{
    EXPECT_EQ(integer_type->id(), "apz");
    EXPECT_EQ(integer_type->name(), "MultiPrecisionInteger");
}


TEST_F(TestIntegerType, TestRefCounting)
{
    EXPECT_EQ(integer_type->ref_count(), 1);

    {
        RPY_MAYBE_UNUSED TypePtr new_ref(integer_type);
        EXPECT_EQ(integer_type->ref_count(), 1);
    }

    EXPECT_EQ(integer_type->ref_count(), 1);
}

TEST_F(TestIntegerType, TestBasicProperties)
{
    EXPECT_GE(integer_type->object_size(), sizeof(void*));

    EXPECT_FALSE(concepts::is_standard_layout(*integer_type));
    EXPECT_FALSE(concepts::is_trivially_copyable(*integer_type));
    EXPECT_FALSE(concepts::is_trivially_constructible(*integer_type));
    EXPECT_FALSE(concepts::is_trivially_default_constructible(*integer_type));
    EXPECT_FALSE(concepts::is_trivially_copy_constructible(*integer_type));
    EXPECT_FALSE(concepts::is_trivially_copy_assignable(*integer_type));
    EXPECT_FALSE(concepts::is_trivially_destructible(*integer_type));
    EXPECT_FALSE(concepts::is_polymorphic(*integer_type));
    EXPECT_TRUE(concepts::is_signed(*integer_type));
    EXPECT_FALSE(concepts::is_unsigned(*integer_type));
    EXPECT_FALSE(concepts::is_integral(*integer_type));
    EXPECT_FALSE(concepts::is_arithmetic(*integer_type));
}


TEST_F(TestIntegerType, TestDisplayAndParseFromString)
{
    Value value(integer_type, string_view("1234567890"));
    std::stringstream ss;
    integer_type->display(ss, value.data());
    EXPECT_EQ(ss.str(), "1234567890");
}


/******************************************************************************
 *                                Comparison                                  *
 ******************************************************************************/

 TEST_F(TestIntegerType, TestHasEqualOperator)
{
    auto* comp = comp_trait();
    ASSERT_NE(comp, nullptr);

    EXPECT_TRUE(comp->has_comparison(ComparisonType::Equal));
}

TEST_F(TestIntegerType, TestHasLessThanOperator)
{
    auto* comp = comp_trait();
    ASSERT_NE(comp, nullptr);
    EXPECT_TRUE(comp->has_comparison(ComparisonType::Less));
}

TEST_F(TestIntegerType, TestHasGreaterThanOperator)
{
    auto* comp = comp_trait();
    ASSERT_NE(comp, nullptr);
    EXPECT_TRUE(comp->has_comparison(ComparisonType::Greater));
}

TEST_F(TestIntegerType, TestHasLessThanOrEqualOperator)
{
    auto* comp = comp_trait();
    ASSERT_NE(comp, nullptr);
    EXPECT_TRUE(comp->has_comparison(ComparisonType::LessEqual));
}

TEST_F(TestIntegerType, TestHasGreaterThanOrEqualOperator)
{
    auto* comp = comp_trait();
    ASSERT_NE(comp, nullptr);
    EXPECT_TRUE(comp->has_comparison(ComparisonType::GreaterEqual));
}

/******************************************************************************
 *                                Arithmetic                                  *
 ******************************************************************************/

TEST_F(TestIntegerType, TestAddOperator)
{
    auto* arith = arith_trait();
    ASSERT_NE(arith, nullptr);

    EXPECT_TRUE(arith->has_operation(ArithmeticOperation::Add));
}

TEST_F(TestIntegerType, TestSubtractOperator)
{
    auto* arith = arith_trait();
    ASSERT_NE(arith, nullptr);

    EXPECT_TRUE(arith->has_operation(ArithmeticOperation::Sub));
}

TEST_F(TestIntegerType, TestMultiplyOperator)
{
    auto* arith = arith_trait();
    ASSERT_NE(arith, nullptr);

    EXPECT_TRUE(arith->has_operation(ArithmeticOperation::Mul));
}

TEST_F(TestIntegerType, TestDivideOperator)
{
    auto* arith = arith_trait();
    ASSERT_NE(arith, nullptr);

    EXPECT_FALSE(arith->has_operation(ArithmeticOperation::Div));
}

/******************************************************************************
 *                                Number-like                                 *
 ******************************************************************************/

TEST_F(TestIntegerType, TestRealAndImaginary)
{
    const auto* num = num_trait();
    ASSERT_NE(num, nullptr);

    EXPECT_TRUE(num->has_function(NumberFunction::Real));
    EXPECT_TRUE(num->has_function(NumberFunction::Imaginary));

}

TEST_F(TestIntegerType, TestAbsFunction)
{
    const auto* num = num_trait();
    ASSERT_NE(num, nullptr);
    EXPECT_TRUE(num->has_function(NumberFunction::Abs));
}

TEST_F(TestIntegerType, TestSqrtExpLogFunction)
{
    // Rational numbers do not have sqrt, exp, or log.
    const auto* num = num_trait();
    ASSERT_NE(num, nullptr);

    EXPECT_FALSE(num->has_function(NumberFunction::Sqrt));
    EXPECT_FALSE(num->has_function(NumberFunction::Exp));
    EXPECT_FALSE(num->has_function(NumberFunction::Log));
}

TEST_F(TestIntegerType, TestPowFunction)
{
    const auto* num = num_trait();
    ASSERT_NE(num, nullptr);
    EXPECT_TRUE(num->has_function(NumberFunction::Pow));
}


TEST_F(TestIntegerType, TestFromRationalFunction)
{
    const auto* num = num_trait();
    ASSERT_NE(num, nullptr);
    EXPECT_TRUE(num->has_function(NumberFunction::FromRational));

}
