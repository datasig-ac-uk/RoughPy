//
// Created by sam on 26/11/24.
//
#include <gtest/gtest.h>

#include "indeterminate.h"


using namespace rpy;
using namespace rpy::generics;


// Test Default Constructor
TEST(TestIndeterminate, TestDefaultConstructor) {
    Indeterminate ind;
    EXPECT_EQ(ind.prefix(), "x");
    EXPECT_EQ(ind.index(), 0);
}

// Test Constructor with Integer
TEST(TestIndeterminate, TestConstructorWithInteger) {
    Indeterminate ind(123);
    EXPECT_EQ(ind.prefix(), "x");
    EXPECT_EQ(ind.index(), 123);
}

// Test Constructor with Character and Integer
TEST(TestIndeterminate, TestConstructorWithCharAndInteger) {
    Indeterminate ind('y', 456);
    EXPECT_EQ(ind.prefix(), "y");
    EXPECT_EQ(ind.index(), 456);
}

// Test Constructor with Character Array and Integer
TEST(TestIndeterminate, TestConstructorWithCharArrayAndInteger) {
    Indeterminate ind({'z'}, 789);
    EXPECT_EQ(ind.prefix(), "z");
    EXPECT_EQ(ind.index(), 789);
}

// Test Equality Operator
TEST(TestIndeterminate, TestEqualityOperator) {
    Indeterminate ind1('x', 303);
    Indeterminate ind2('x', 303);
    EXPECT_TRUE(ind1 == ind2);

    Indeterminate ind3('y', 303);
    EXPECT_FALSE(ind1 == ind3);

    Indeterminate ind4('x', 304);
    EXPECT_FALSE(ind1 == ind4);

    Indeterminate ind5('x', 2304134);
    EXPECT_FALSE(ind1 == ind5);
}

// Test Inequality Operator
TEST(TestIndeterminate, TestInequalityOperator) {
    Indeterminate ind1('x', 304);
    Indeterminate ind2('x', 304);
    EXPECT_FALSE(ind1 != ind2);

    Indeterminate ind3('y', 304);
    EXPECT_TRUE(ind1 != ind3);

    Indeterminate ind4('x', 305);
    EXPECT_TRUE(ind1 != ind4);

    Indeterminate ind5('x', 2304134);
    EXPECT_TRUE(ind1 != ind5);
}

// Test Less Than Operator
TEST(TestIndeterminate, TestLessThanOperator) {
    Indeterminate ind1('x', 606);
    Indeterminate ind2('x', 707);
    EXPECT_LT(ind1, ind2);
}

// Test Less Than or Equal Operator
TEST(TestIndeterminate, TestLessThanOrEqualOperator) {
    Indeterminate ind1('x', 808);
    Indeterminate ind2('x', 808);
    EXPECT_LE(ind1, ind2);
}

// Test Greater Than Operator
TEST(TestIndeterminate, TestGreaterThanOperator) {
    Indeterminate ind1('x', 909);
    Indeterminate ind2('x', 808);
    EXPECT_GT(ind1, ind2);
}

// Test Greater Than or Equal Operator
TEST(TestIndeterminate, TestGreaterThanOrEqualOperator) {
    Indeterminate ind1('x', 1010);
    Indeterminate ind2('x', 1010);
    EXPECT_GE(ind1, ind2);
}

TEST(TestIndeterminate, TestHash)
{
    Indeterminate ind1('x', 1);
    Indeterminate ind2('x', 1);
    EXPECT_EQ(ind1, ind2);
    EXPECT_EQ(hash_value(ind1), hash_value(ind2));

    Indeterminate ind3('x', 2);
    EXPECT_NE(hash_value(ind1), hash_value(ind3));
}


TEST(TestIndeterminate, TestDisplay)
{
    Indeterminate ind1('x', 1);

    std::stringstream ss1;
    ss1 << ind1;
    EXPECT_EQ(ss1.str(), "x1");
}
