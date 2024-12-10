//
// Created by sammorley on 18/11/24.
//

#include <cmath>
#include <typeinfo>

#include <gtest/gtest.h>

#include <roughpy/generics/arithmetic_trait.h>
#include <roughpy/generics/comparison_trait.h>
#include <roughpy/generics/number_trait.h>
#include <roughpy/generics/type.h>

using namespace rpy;
using namespace rpy::generics;

TEST(TestFloatType, TestID)
{
    const auto float_type = get_type<float>();
    EXPECT_EQ(float_type->id(), "f32");
}

TEST(TestFloatType, TestName)
{
    const auto float_type = get_type<float>();
    EXPECT_EQ(float_type->name(), "float");
}

TEST(TestFloatType, TestTypeInfo)
{
    const auto type = get_type<float>();
    EXPECT_EQ(type->type_info(), typeid(float));
}

TEST(TestFloatType, TestHash)
{
    const auto type = get_type<float>();

    Hash<float> hasher;
    for (float val : {1.f, -2.f, 3.141592653589793f, -2.7182818284598f}) {
        EXPECT_EQ(type->hash_of(&val), hasher(val));
    }
}
TEST(TestFloatType, ParseFromStringValidString)
{
    const auto type = get_type<float>();

    string_view str = "3.141592653589793";
    float value;
    auto result = type->parse_from_string(&value, str);

    ASSERT_TRUE(result);
    EXPECT_EQ(value, 3.141592653589793f);

}

TEST(TestFloatType, TestParseFromStringInvalidString)
{
    const auto type = get_type<float>();
    float value;

    string_view str = "bad value";
    auto result = type->parse_from_string(&value, str);

    EXPECT_FALSE(result);
}

/******************************************************************************
 *                                Comparison                                  *
 ******************************************************************************/

TEST(TestFloatType, TestHasEqualOperator) {
    const auto type = get_type<float>();
    ComparisonTraitImpl<float> traits(type.get());
    EXPECT_TRUE(traits.has_comparison(ComparisonType::Equal));
}

TEST(TestFloatType, TestHasLessThanOperator)
{
    const auto type = get_type<float>();
    ComparisonTraitImpl<float> traits(type.get());
    EXPECT_TRUE(traits.has_comparison(ComparisonType::Less));
}
TEST(TestFloatType, TestHasLessThanOrEqualOperator)
{
    const auto type = get_type<float>();
    ComparisonTraitImpl<float> traits(type.get());
    EXPECT_TRUE(traits.has_comparison(ComparisonType::LessEqual));
}
TEST(TestFloatType, TestHasGreaterThanOperator)
{
    const auto type = get_type<float>();
    ComparisonTraitImpl<float> traits(type.get());
    EXPECT_TRUE(traits.has_comparison(ComparisonType::Greater));
}
TEST(TestFloatType, TestHasGreaterThanOrEqualOperator)
{
    const auto type = get_type<float>();
    ComparisonTraitImpl<float> traits(type.get());
    EXPECT_TRUE(traits.has_comparison(ComparisonType::GreaterEqual));
}

/******************************************************************************
 *                                Arithmetic                                  *
 ******************************************************************************/

TEST(TestFloatType, TestAddOperator)
{
    const auto type = get_type<float>();
    ArithmeticTraitImpl<float> traits(type.get(), type.get());
    ASSERT_TRUE(traits.has_operation(ArithmeticOperation::Add));

    float lhs = 2.0f;
    float rhs = -3.0f;

    traits.unsafe_add_inplace(&lhs, &rhs);

    EXPECT_FLOAT_EQ(lhs, 2.0f + rhs);
}

TEST(TestFloatType, TestSubOperator)
{
    const auto type = get_type<float>();
    ArithmeticTraitImpl<float> traits(type.get(), type.get());
    ASSERT_TRUE(traits.has_operation(ArithmeticOperation::Sub));

    float lhs = 2.0f;
    float rhs = -3.0f;

    traits.unsafe_sub_inplace(&lhs, &rhs);

    EXPECT_FLOAT_EQ(lhs, 2.0f - rhs);
}

TEST(TestFloatType, TestMulOperator)
{
    const auto type = get_type<float>();
    ArithmeticTraitImpl<float> traits(type.get(), type.get());
    ASSERT_TRUE(traits.has_operation(ArithmeticOperation::Mul));

    float lhs = 2.0f;
    float rhs = -3.0f;

    traits.unsafe_mul_inplace(&lhs, &rhs);

    EXPECT_FLOAT_EQ(lhs, 2.0f * rhs);
}

TEST(TestFloatType, TestDivOperator)
{
    const auto type = get_type<float>();
    ArithmeticTraitImpl<float> traits(type.get(), type.get());
    ASSERT_TRUE(traits.has_operation(ArithmeticOperation::Div));

    float lhs = 2.0f;
    float rhs = -3.0f;

    traits.unsafe_div_inplace(&lhs, &rhs);

    EXPECT_FLOAT_EQ(lhs, 2.0f / rhs);
}







/******************************************************************************
 *                                Number-like                                 *
 ******************************************************************************/

TEST(TestFloatType, TestRealAndImaginary)
{
    const auto type = get_type<float>();
    NumberTraitImpl<float> traits(type.get());

    ASSERT_TRUE(traits.has_function(NumberFunction::Real));
    ASSERT_TRUE(traits.has_function(NumberFunction::Imaginary));

    float value = 2.0f;
    float real_part, imaginary_part;

    traits.unsafe_real(&real_part, &value);
    traits.unsafe_imaginary(&imaginary_part, &value);

    EXPECT_EQ(value, real_part);
    EXPECT_EQ(imaginary_part, 0.0f);
}


TEST(TestFloatType, TestAbsFunction)
{
    const auto type = get_type<float>();
    NumberTraitImpl<float> traits(type.get());
    ASSERT_TRUE(traits.has_function(NumberFunction::Abs));

    float value = -2.0f;
    float abs_value;
    traits.unsafe_abs(&abs_value, &value);

    EXPECT_EQ(abs_value, -value);
}

TEST(TestFloatType, TestSqrtFunction)
{
    const auto type = get_type<float>();
    NumberTraitImpl<float> traits(type.get());
    ASSERT_TRUE(traits.has_function(NumberFunction::Sqrt));

    float value = 9.0f;
    float sqrt;
    traits.unsafe_sqrt(&sqrt, &value);

    EXPECT_EQ(sqrt, std::sqrt(value));
}

TEST(TestFloatType, TestPowFunction)
{
    const auto type = get_type<float>();
    NumberTraitImpl<float> traits(type.get());
    ASSERT_TRUE(traits.has_function(NumberFunction::Pow));

    float value = 2.0f;
    exponent_t exponent = 4;
    float pow_value;
    traits.unsafe_pow(&pow_value, &value, exponent);

    EXPECT_EQ(pow_value, std::pow(value, exponent));
}

TEST(TestFloatType, TestExpFunction)
{
    const auto type = get_type<float>();
    NumberTraitImpl<float> traits(type.get());
    ASSERT_TRUE(traits.has_function(NumberFunction::Exp));

    float value = 2.0;
    float result;
    traits.unsafe_exp(&result, &value);

    EXPECT_FLOAT_EQ(result, std::exp(value));
}

TEST(TestFloatType, TestLogFunction)
{
    const auto type = get_type<float>();
    NumberTraitImpl<float> traits(type.get());
    ASSERT_TRUE(traits.has_function(NumberFunction::Log));

    float value = 2.0;
    float result;
    traits.unsafe_log(&result, &value);
    EXPECT_FLOAT_EQ(result, std::log(value));
}

TEST(TestFloatType, TestFromRationalFunction)
{
    const auto type = get_type<float>();
    NumberTraitImpl<float> traits(type.get());
    ASSERT_TRUE(traits.has_function(NumberFunction::FromRational));

    float result;
    int64_t numerator = 243;
    int64_t denominator = 192;

    traits.unsafe_from_rational(&result, numerator, denominator);

    EXPECT_FLOAT_EQ(result, static_cast<float>(numerator) / denominator);
}

