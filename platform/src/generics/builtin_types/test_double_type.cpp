//
// Created by sam on 17/11/24.
//

#include <typeinfo>

#include <gtest/gtest.h>

#include <roughpy/generics/arithmetic_trait.h>
#include <roughpy/generics/comparison_trait.h>
#include <roughpy/generics/number_trait.h>
#include <roughpy/generics/type.h>



using namespace rpy;
using namespace rpy::generics;

TEST(TestDoubleType, TestID)
{
    const auto type = get_type<double>();
    EXPECT_EQ(type->id(), "f64");
}

TEST(TestDoubleType, TestTypeInfo)
{
    const auto type = get_type<double>();
    EXPECT_EQ(type->type_info(), typeid(double));
}

/******************************************************************************
 *                                Comparison                                  *
 ******************************************************************************/

TEST(TestDoubleType, TestHasEqualOperator)
{
    const auto double_type = get_type<double>();
    ComparisonTraitImpl<double> traits(double_type.get());
    EXPECT_TRUE(traits.has_comparison(ComparisonType::Equal));
}

TEST(TestDoubleType, TestHasLessOperator)
{
    const auto double_type = get_type<double>();
    ComparisonTraitImpl<double> traits(double_type.get());
    EXPECT_TRUE(traits.has_comparison(ComparisonType::Less));
}

TEST(TestDoubleType, TestHasLessEqualOperator)
{
    const auto double_type = get_type<double>();
    ComparisonTraitImpl<double> traits(double_type.get());
    EXPECT_TRUE(traits.has_comparison(ComparisonType::LessEqual));
}

TEST(TestDoubleType, TestHasGreaterOperator)
{
    const auto double_type = get_type<double>();
    ComparisonTraitImpl<double> traits(double_type.get());
    EXPECT_TRUE(traits.has_comparison(ComparisonType::Greater));
}

TEST(TestDoubleType, TestHasGreaterEqualOperator)
{
    const auto double_type = get_type<double>();
    ComparisonTraitImpl<double> traits(double_type.get());
    EXPECT_TRUE(traits.has_comparison(ComparisonType::GreaterEqual));
}


/******************************************************************************
 *                                Arithmetic                                  *
 ******************************************************************************/

TEST(TestDoubleType, TestAddOperator)
{
    const auto type = get_type<double>();
    ArithmeticTraitImpl<double> traits(type.get(), type.get());
    ASSERT_TRUE(traits.has_operation(ArithmeticOperation::Add));

    double lhs = 2.0;
    double rhs = -3.0;

    traits.unsafe_add_inplace(&lhs, &rhs);

    EXPECT_DOUBLE_EQ(lhs, 2.0 + rhs);
}

TEST(TestDoubleType, TestSubOperator)
{
    const auto type = get_type<double>();
    ArithmeticTraitImpl<double> traits(type.get(), type.get());
    ASSERT_TRUE(traits.has_operation(ArithmeticOperation::Sub));

    double lhs = 2.0;
    double rhs = -3.0;

    traits.unsafe_sub_inplace(&lhs, &rhs);

    EXPECT_DOUBLE_EQ(lhs, 2.0 - rhs);
}

TEST(TestDoubleType, TestMulOperator)
{
    const auto type = get_type<double>();
    ArithmeticTraitImpl<double> traits(type.get(), type.get());
    ASSERT_TRUE(traits.has_operation(ArithmeticOperation::Mul));

    double lhs = 2.0;
    double rhs = -3.0;

    traits.unsafe_mul_inplace(&lhs, &rhs);

    EXPECT_DOUBLE_EQ(lhs, 2.0 * rhs);
}

TEST(TestDoubleType, TestDivOperator)
{
    const auto type = get_type<double>();
    ArithmeticTraitImpl<double> traits(type.get(), type.get());
    ASSERT_TRUE(traits.has_operation(ArithmeticOperation::Div));

    double lhs = 2.0;
    double rhs = -3.0;

    traits.unsafe_div_inplace(&lhs, &rhs);

    EXPECT_DOUBLE_EQ(lhs, 2.0 / rhs);
}

/******************************************************************************
 *                                Number-like                                 *
 ******************************************************************************/

TEST(TestDoubleType, TestRealAndImaginary)
{
    const auto type = get_type<double>();
    NumberTraitImpl<double> traits(type.get());

    ASSERT_TRUE(traits.has_function(NumberFunction::Real));
    ASSERT_TRUE(traits.has_function(NumberFunction::Imaginary));

    double value = 2.0;
    double real_part, imaginary_part;

    traits.unsafe_real(&real_part, &value);
    traits.unsafe_imaginary(&imaginary_part, &value);

    EXPECT_EQ(value, real_part);
    EXPECT_EQ(imaginary_part, 0.0);
}

TEST(TestDoubleType, TestAbsFunction)
{
    const auto type = get_type<double>();
    NumberTraitImpl<double> traits(type.get());
    ASSERT_TRUE(traits.has_function(NumberFunction::Abs));

    double value = -2.0;
    double abs_value;
    traits.unsafe_abs(&abs_value, &value);

    EXPECT_EQ(abs_value, -value);
}

TEST(TestDoubleType, TestHasSqrtFunction)
{
    const auto type = get_type<double>();
    NumberTraitImpl<double> traits(type.get());
    ASSERT_TRUE(traits.has_function(NumberFunction::Sqrt));

    double value = 9.0;
    double sqrt;
    traits.unsafe_sqrt(&sqrt, &value);

    EXPECT_EQ(sqrt, std::sqrt(value));
}

TEST(TestDoubleType, TestHasPowFunction)
{
    const auto type = get_type<double>();
    NumberTraitImpl<double> traits(type.get());
    ASSERT_TRUE(traits.has_function(NumberFunction::Pow));

    double value = 2.0;
    exponent_t exponent = 4;
    double pow_value;
    traits.unsafe_pow(&pow_value, &value, exponent);

    EXPECT_EQ(pow_value, std::pow(value, exponent));
}

TEST(TestDoubleType, TestHasExpFunction)
{
    const auto type = get_type<double>();
    NumberTraitImpl<double> traits(type.get());
    ASSERT_TRUE(traits.has_function(NumberFunction::Exp));

    double value = 2.0;
    double result;
    traits.unsafe_exp(&result, &value);

    EXPECT_DOUBLE_EQ(result, std::exp(value));
}

TEST(TestDoubleType, TestHasLogFunction)
{
    const auto type = get_type<double>();
    NumberTraitImpl<double> traits(type.get());
    ASSERT_TRUE(traits.has_function(NumberFunction::Log));

    double value = 2.0;
    double result;
    traits.unsafe_log(&result, &value);
    EXPECT_DOUBLE_EQ(result, std::log(value));
}

TEST(TestDoubleType, TestHasFromRationalFunction)
{
    const auto type = get_type<double>();
    NumberTraitImpl<double> traits(type.get());
    ASSERT_TRUE(traits.has_function(NumberFunction::FromRational));

    double result;
    int64_t numerator = 243;
    int64_t denominator = 192;

    traits.unsafe_from_rational(&result, numerator, denominator);

    EXPECT_FLOAT_EQ(result, static_cast<double>(numerator) / denominator);
}

