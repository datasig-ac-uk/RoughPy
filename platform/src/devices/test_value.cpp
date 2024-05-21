//
// Created by sam on 5/21/24.
//

#include <roughpy/devices/value.h>

#include <gtest/gtest.h>

using namespace rpy;
using namespace rpy::devices;

namespace {

class TestValue : public testing::Test
{
protected:
    TestValue() : double_type(get_type("f64")) {}

    const Type* double_type;
};

}// namespace

TEST_F(TestValue, TestAddInplace)
{
    Value left(0.5);
    Value right(1.53);

    left += right;
    EXPECT_EQ(left, Value(0.5 + 1.53));
}

TEST_F(TestValue, TestSubtractInplace)
{
    Value left(3.0);
    Value right(1.5);

    left -= right;
    EXPECT_EQ(left, Value(3.0 - 1.5));
}

TEST_F(TestValue, TestMultiplyInplace)
{
    Value left(7.0);
    Value right(2.0);

    left *= right;
    EXPECT_EQ(left, Value(7.0 * 2.0));
}

TEST_F(TestValue, TestDivideInplace)
{
    Value left(10.0);
    Value right(2.0);

    left /= right;
    EXPECT_EQ(left, Value(10.0 / 2.0));
}

TEST_F(TestValue, TestAdd)
{
    Value left(5.0);
    Value right(10.0);

    Value result = left + right;
    EXPECT_EQ(result, Value(5.0 + 10.0));
}

TEST_F(TestValue, TestSubtract)
{
    Value left(20.0);
    Value right(5.0);

    Value result = left - right;
    EXPECT_EQ(result, Value(20.0 - 5.0));
}

TEST_F(TestValue, TestMultiply)
{
    Value left(3.0);
    Value right(6.0);

    Value result = left * right;
    EXPECT_EQ(result, Value(3.0 * 6.0));
}

TEST_F(TestValue, TestDivide)
{
    Value left(12.0);
    Value right(4.0);

    Value result = left / right;
    EXPECT_EQ(result, Value(12.0 / 4.0));
}

TEST_F(TestValue, TestChangeType)
{
    Value value(15.0);
    ASSERT_EQ(value.type(), double_type);

    const auto* i32_type = get_type("i32");
    value.change_type(i32_type);
    EXPECT_EQ(value.type(), i32_type);
    EXPECT_EQ(value, Value(15));
}

TEST_F(TestValue, TestAssignFromInt)
{
    Value value(double_type);

    value = 15;

    EXPECT_EQ(value.type(), double_type);
    EXPECT_EQ(value, Value(15.0));
}

TEST_F(TestValue, AssignIntFromFloat)
{
    Value value(get_type("i32"));

    value = 15.5;

    EXPECT_EQ(value, Value(15));
}

TEST_F(TestValue, AssignFloatToEmpty)
{
    Value value;
    value = 15.5;

    EXPECT_EQ(value.type(), double_type);
    EXPECT_EQ(value, Value(15.5));
}

TEST_F(TestValue, TestLessThan)
{
    Value smaller(10.0);
    Value larger(20.0);
    EXPECT_TRUE(smaller < larger);
}

TEST_F(TestValue, TestLessThanOrEqual)
{
    Value smallerOrEqual1(10.0);
    Value smallerOrEqual2(10.0);
    Value larger(20.0);
    EXPECT_TRUE(smallerOrEqual1 <= larger);
    EXPECT_TRUE(smallerOrEqual2 <= smallerOrEqual2);
}

TEST_F(TestValue, TestGreaterThan)
{
    Value larger(20.0);
    Value smaller(10.0);
    EXPECT_TRUE(larger > smaller);
}

TEST_F(TestValue, TestGreaterThanOrEqual)
{
    Value largerOrEqual1(20.0);
    Value largerOrEqual2(20.0);
    Value smaller(10.0);
    EXPECT_TRUE(largerOrEqual1 >= smaller);
    EXPECT_TRUE(largerOrEqual2 >= largerOrEqual2);
}

TEST_F(TestValue, TestEqual)
{
    Value val1(10.0);
    Value val2(10.0);
    EXPECT_TRUE(val1 == val2);
}

TEST_F(TestValue, TestNotEqual)
{
    Value val1(10.0);
    Value val2(20.0);
    EXPECT_TRUE(val1 != val2);
}

TEST_F(TestValue, TestStreamOutOperator)
{
    Value testValue(5.5);
    std::ostringstream oss;

    oss << testValue;

    EXPECT_EQ(oss.str(), "5.5");
}

TEST_F(TestValue, TestAddInplaceInteger)
{
    Value left(0.5);
    int right = 1;

    left += right;
    EXPECT_EQ(left, Value(0.5 + 1));
}

TEST_F(TestValue, TestSubtractInplaceInteger)
{
    Value left(3.0);
    int right = 1;

    left -= right;
    EXPECT_EQ(left, Value(3.0 - 1));
}

TEST_F(TestValue, TestMultiplyInplaceInteger)
{
    Value left(7.0);
    int right = 2;

    left *= right;
    EXPECT_EQ(left, Value(7.0 * 2));
}

TEST_F(TestValue, TestDivideInplaceInteger)
{
    Value left(10.0);
    int right = 2;

    left /= right;
    EXPECT_EQ(left, Value(10.0 / 2));
}
