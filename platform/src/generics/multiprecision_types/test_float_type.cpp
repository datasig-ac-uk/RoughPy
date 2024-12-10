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

class TestMPFloatType : public ::testing::Test
{
public:
    const size_t precision = 75;
protected:

    void SetUp() override
    {
        float_type = MultiPrecisionTypes::get().float_type(precision);
        if (float_type == nullptr) {
            FAIL() << "Failed to create float type with precision " << precision;
        }
    }

public:
    const ComparisonTrait* comp_trait() const noexcept
    {
        return trait_cast<const ComparisonTrait>(
                float_type->get_builtin_trait(BuiltinTraitID::Comparison)
        );
    }

    const ArithmeticTrait* arith_trait() const noexcept
    {
        return trait_cast<const ArithmeticTrait>(
                float_type->get_builtin_trait(BuiltinTraitID::Arithmetic)
        );
    }

    const NumberTrait* num_trait() const noexcept
    {
        return trait_cast<const NumberTrait>(
                float_type->get_builtin_trait(BuiltinTraitID::Number)
        );
    }


    TypePtr float_type;

};


}




TEST_F(TestMPFloatType, TestID)
{
    EXPECT_EQ(float_type->id(), "apf");
    EXPECT_EQ(float_type->name(), "MultiPrecisionFloat");
}

TEST_F(TestMPFloatType, TestRefCounting)
{
        EXPECT_EQ(float_type->ref_count(), 1);

    {
        RPY_MAYBE_UNUSED TypePtr new_ref(float_type);
        EXPECT_EQ(float_type->ref_count(), 2);
    }

    EXPECT_EQ(float_type->ref_count(), 1);
}

TEST_F(TestMPFloatType, TestBasicProperties)
{
    EXPECT_GE(float_type->object_size(), sizeof(void*));

    EXPECT_FALSE(concepts::is_standard_layout(*float_type));
    EXPECT_FALSE(concepts::is_trivially_copyable(*float_type));
    EXPECT_FALSE(concepts::is_trivially_constructible(*float_type));
    EXPECT_FALSE(concepts::is_trivially_default_constructible(*float_type));
    EXPECT_FALSE(concepts::is_trivially_copy_constructible(*float_type));
    EXPECT_FALSE(concepts::is_trivially_copy_assignable(*float_type));
    EXPECT_FALSE(concepts::is_trivially_destructible(*float_type));
    EXPECT_FALSE(concepts::is_polymorphic(*float_type));
    EXPECT_TRUE(concepts::is_signed(*float_type));
    EXPECT_FALSE(concepts::is_unsigned(*float_type));
    EXPECT_FALSE(concepts::is_integral(*float_type));
    EXPECT_FALSE(concepts::is_arithmetic(*float_type));
}


TEST_F(TestMPFloatType, TestDisplayAndParseFromString)
{
    Value value(float_type, string_view("1234567890"));
    std::stringstream ss;
    float_type->display(ss, value.data());
    EXPECT_EQ(ss.str(), "1234567890");
}


/******************************************************************************
 *                                Comparison                                  *
 ******************************************************************************/

TEST_F(TestMPFloatType, TestHasEqualOperator)
{
    auto* comp = comp_trait();
    ASSERT_NE(comp, nullptr);

    EXPECT_TRUE(comp->has_comparison(ComparisonType::Equal));
}

TEST_F(TestMPFloatType, TestHasLessThanOperator)
{
    auto* comp = comp_trait();
    ASSERT_NE(comp, nullptr);
    EXPECT_TRUE(comp->has_comparison(ComparisonType::Less));
}

TEST_F(TestMPFloatType, TestHasGreaterThanOperator)
{
    auto* comp = comp_trait();
    ASSERT_NE(comp, nullptr);
    EXPECT_TRUE(comp->has_comparison(ComparisonType::Greater));
}

TEST_F(TestMPFloatType, TestHasLessThanOrEqualOperator)
{
    auto* comp = comp_trait();
    ASSERT_NE(comp, nullptr);
    EXPECT_TRUE(comp->has_comparison(ComparisonType::LessEqual));
}

TEST_F(TestMPFloatType, TestHasGreaterThanOrEqualOperator)
{
    auto* comp = comp_trait();
    ASSERT_NE(comp, nullptr);
    EXPECT_TRUE(comp->has_comparison(ComparisonType::GreaterEqual));
}

/******************************************************************************
 *                                Arithmetic                                  *
 ******************************************************************************/

TEST_F(TestMPFloatType, TestAddOperator)
{
    auto* arith = arith_trait();
    ASSERT_NE(arith, nullptr);

    EXPECT_TRUE(arith->has_operation(ArithmeticOperation::Add));
}

TEST_F(TestMPFloatType, TestSubtractOperator)
{
    auto* arith = arith_trait();
    ASSERT_NE(arith, nullptr);

    EXPECT_TRUE(arith->has_operation(ArithmeticOperation::Sub));
}

TEST_F(TestMPFloatType, TestMultiplyOperator)
{
    auto* arith = arith_trait();
    ASSERT_NE(arith, nullptr);

    EXPECT_TRUE(arith->has_operation(ArithmeticOperation::Mul));
}

TEST_F(TestMPFloatType, TestDivideOperator)
{
    auto* arith = arith_trait();
    ASSERT_NE(arith, nullptr);

    EXPECT_TRUE(arith->has_operation(ArithmeticOperation::Div));
}

/******************************************************************************
 *                                Number-like                                 *
 ******************************************************************************/

TEST_F(TestMPFloatType, TestRealAndImaginary)
{
    const auto* num = num_trait();
    ASSERT_NE(num, nullptr);

    EXPECT_TRUE(num->has_function(NumberFunction::Real));
    EXPECT_TRUE(num->has_function(NumberFunction::Imaginary));

}

TEST_F(TestMPFloatType, TestAbsFunction)
{
    const auto* num = num_trait();
    ASSERT_NE(num, nullptr);
    EXPECT_TRUE(num->has_function(NumberFunction::Abs));
}

TEST_F(TestMPFloatType, TestSqrtFunction)
{
    const auto* num = num_trait();
    ASSERT_NE(num, nullptr);

    EXPECT_TRUE(num->has_function(NumberFunction::Sqrt));
}

TEST_F(TestMPFloatType, TestExpFunction)
{
    const auto* num = num_trait();
    ASSERT_NE(num, nullptr);
    EXPECT_TRUE(num->has_function(NumberFunction::Exp));
}

TEST_F(TestMPFloatType, TestLogFunction)
{
    const auto* num = num_trait();
    ASSERT_NE(num, nullptr);
    EXPECT_TRUE(num->has_function(NumberFunction::Log));
}

TEST_F(TestMPFloatType, TestPowFunction)
{
    const auto* num = num_trait();
    ASSERT_NE(num, nullptr);
    EXPECT_TRUE(num->has_function(NumberFunction::Pow));
}


TEST_F(TestMPFloatType, TestFromRationalFunction)
{
    const auto* num = num_trait();
    ASSERT_NE(num, nullptr);
    EXPECT_TRUE(num->has_function(NumberFunction::FromRational));

}


