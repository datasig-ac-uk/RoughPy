//
// Created by sam on 15/11/24.
//

#include <cmath>

#include <gtest/gtest.h>

#include "roughpy/core/hash.h"
#include "roughpy/generics/type.h"
#include "roughpy/generics/values.h"

using namespace rpy;
using namespace rpy::generics;


TEST(TestValue, TestConstructFromString)
{
    Value v(get_type<double>(), string_view("3.1415"));

    EXPECT_EQ(v, Value(3.1415));
}


TEST(TestValue, TestAddInplace)
{
    Value left(0.5);
    Value right(1.53);

    left += right;
    EXPECT_EQ(left, Value(0.5 + 1.53));
}

TEST(TestValue, TestSubtractInplace)
{
    Value left(3.0);
    Value right(1.5);

    left -= right;
    EXPECT_EQ(left, Value(3.0 - 1.5));
}

TEST(TestValue, TestMultiplyInplace)
{
    Value left(7.0);
    Value right(2.0);

    left *= right;
    EXPECT_EQ(left, Value(7.0 * 2.0));
}

TEST(TestValue, TestDivideInplace)
{
    Value left(10.0);
    Value right(2.0);

    left /= right;
    EXPECT_EQ(left, Value(10.0 / 2.0));
}

TEST(TestValue, TestAdd)
{
    Value left(5.0);
    Value right(10.0);

    Value result = left + right;
    EXPECT_EQ(result, Value(5.0 + 10.0));
}

TEST(TestValue, TestSubtract)
{
    Value left(20.0);
    Value right(5.0);

    Value result = left - right;
    EXPECT_EQ(result, Value(20.0 - 5.0));
}

TEST(TestValue, TestMultiply)
{
    Value left(3.0);
    Value right(6.0);

    Value result = left * right;
    EXPECT_EQ(result, Value(3.0 * 6.0));
}

TEST(TestValue, TestDivide)
{
    Value left(12.0);
    Value right(4.0);

    Value result = left / right;
    EXPECT_EQ(result, Value(12.0 / 4.0));
}

TEST(TestValue, TestEqualityTrue)
{
    Value left(2.0);
    Value right(2.0);

    EXPECT_TRUE(left == right);
}

TEST(TestValue, TestEqualityFalse)
{
    Value left(2.0);
    Value right(1.0);

    EXPECT_FALSE(left == right);
}

TEST(TestValue, TestLessTrue)
{
    Value left(1.0);
    Value right(2.0);

    EXPECT_TRUE(left < right);
}

TEST(TestValue, TestLessFalse)
{
    Value left(2.0);
    Value right(1.0);

    EXPECT_FALSE(left < right);
}

TEST(TestValue, TestLessOrEqualTrue)
{
    Value left(1.0);
    Value right(2.0);
    EXPECT_TRUE(left <= right);
}
TEST(TestValue, TestLessOrEqualFalse)
{
    Value left(2.0);
    Value right(1.0);
    EXPECT_FALSE(left <= right);
}
TEST(TestValue, TestGreaterTrue)
{
    Value left(3.0);
    Value right(1.0);
    EXPECT_TRUE(left > right);
}
TEST(TestValue, TestGreaterFalse)
{
    Value left(1.0);
    Value right(3.0);
    EXPECT_FALSE(left > right);
}
TEST(TestValue, TestGreaterOrEqualTrue)
{
    Value left(2.0);
    Value right(1.0);
    EXPECT_TRUE(left >= right);
}
TEST(TestValue, TestGreaterOrEqualFalse)
{
    Value left(1.0);
    Value right(2.0);
    EXPECT_FALSE(left >= right);
}

TEST(TestValue, TestHash)
{
    double dbl_value = 0.519345234532435;
    Value value(dbl_value);

    Hash<Value> value_hasher;
    Hash<double> dbl_hasher;

    EXPECT_EQ(value_hasher(value), dbl_hasher(dbl_value));
}

TEST(TestValue, TestRealPart)
{
    Value value(2.0);
    Value expected(2.0);

    EXPECT_EQ(math::real(value), expected);
}

TEST(TestValue, TestSqrt)
{
    Value value(16.0);
    Value expected(4.0);

    EXPECT_EQ(generics::math::sqrt(value), expected);
}

TEST(TestValue, TestPow)
{
    Value base(3.0);
    exponent_t exponent = 2;
    Value expected(9.0);

    EXPECT_EQ(generics::math::pow(base, exponent), expected);
}

TEST(TestValue, TestExp)
{
    Value value(1.0);
    Value expected(std::exp(1.0));
    EXPECT_EQ(generics::math::exp(value), expected);
}

TEST(TestValue, TestLog)
{
    Value value(std::exp(1.0));
    Value expected(1.0);
    EXPECT_EQ(generics::math::log(value), expected);
}

// TEST(TestValue, TestSin)
// {
//     Value value(M_PI / 2);
//     Value expected(1.0);
//
//     EXPECT_EQ(generics::math::sin(value), expected);
// }
//
// TEST(TestValue, TestCos)
// {
//     Value value(0.0);
//     Value expected(1.0);
//
//     EXPECT_EQ(generics::math::cos(value), expected);
// }
//
// TEST(TestValue, TestTan)
// {
//     Value value(M_PI / 4);
//     Value expected(1.0);
//
//     EXPECT_EQ(generics::math::tan(value), expected);
// }
//
// TEST(TestValue, TestAsin)
// {
//     Value value(1.0);
//     Value expected(M_PI / 2);
//
//     EXPECT_EQ(generics::math::asin(value), expected);
// }
//
// TEST(TestValue, TestAcos)
// {
//     Value value(1.0);
//     Value expected(0.0);
//
//     EXPECT_EQ(generics::math::acos(value), expected);
// }
//
// TEST(TestValue, TestAtan)
// {
//     Value value(1.0);
//     Value expected(M_PI / 4);
//
//     EXPECT_EQ(generics::math::atan(value), expected);
// }
//
