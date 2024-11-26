//
// Created by sam on 26/11/24.
//

#include <sstream>

#include <gtest/gtest.h>


#include "monomial.h"

using namespace rpy;
using namespace rpy::generics;

TEST(TestMonomial, TestEmptyMonomial)
{
    Monomial m;

    EXPECT_EQ(m.degree(), 0);
    EXPECT_EQ(m.type(), 0);
}

TEST(TestMonomial, TestUnidimConstructorDegreeAndType)
{
    Monomial m(Indeterminate('x', 1));

    EXPECT_EQ(m.degree(), 1);
    EXPECT_EQ(m.type(), 1);
}

TEST(TestMonomial, TestDisplay)
{
    Monomial m1;
    Monomial m2(Indeterminate('x', 1));
    Monomial m3(Indeterminate('x', 2), 2);

    std::stringstream ss1;
    ss1 << m1;

    EXPECT_EQ(ss1.str(), "");

    std::stringstream ss2;
    ss2 << m2;
    EXPECT_EQ(ss2.str(), "x1");

    std::stringstream ss3;
    ss3 << m3;
    EXPECT_EQ(ss3.str(), "x2^2");
}



