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

// TEST(TestValue, TestRealPart)
// {
//     Value value(2.0);
//     Value expected(2.0);
//
//     EXPECT_EQ(math::real(value))
// }

