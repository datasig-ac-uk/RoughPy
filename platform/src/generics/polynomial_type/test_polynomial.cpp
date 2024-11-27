//
// Created by sam on 26/11/24.
//

#include <gtest/gtest.h>

#include "polynomial.h"

using namespace rpy;
using namespace rpy::generics;


namespace rpy::generics {

void PrintTo(const Polynomial& p, std::ostream* os)
{
    poly_print(*os, p);
}

}

TEST(TestPolynomial, TestDefaultValue)
{
    Polynomial p;

    EXPECT_TRUE(p.empty());
    EXPECT_EQ(p.degree(), 0);
    EXPECT_TRUE(poly_cmp_is_zero(p));
}


TEST(TestPolynomial, TestPolynomialEquality)
{
    Polynomial p1{
            {Monomial(), {1, 1}},
            {Monomial(Indeterminate('x'), 1), {2, 1}}
    };// { 1 + 2(x1) }

    Polynomial p2{
            {Monomial(), {1, 1}},
            {Monomial(Indeterminate('x'), 1), {2, 1}}
    }; // { 1 + 2(x1) }

    Polynomial p3{
            {Monomial(), {1, 1}},
            {Monomial(Indeterminate('x'), 1), {3, 1}}
    }; // { 1 + 3(x1) }

    Polynomial p4{
            {Monomial(), {1, 1}},
            {Monomial(Indeterminate('y'), 1), {2, 1}}
    }; // { 1 + 3(x1) }

    EXPECT_TRUE(poly_cmp_equal(p1, p2));
    EXPECT_FALSE(poly_cmp_equal(p1, p3));
    EXPECT_FALSE(poly_cmp_equal(p1, p4));
}

TEST(TestPolynomial, TestPolynomialHash)
{
    Polynomial p1{
            {Monomial(), {1, 1}},
            {Monomial(Indeterminate('x'), 1), {2, 1}}
    };// { 1 + 2(x1) }

    Polynomial p2{
            {Monomial(), {1, 1}},
            {Monomial(Indeterminate('x'), 1), {2, 1}}
    };// { 1 + 2(x1) }

    Polynomial p3{
            {Monomial(), {1, 1}},
            {Monomial(Indeterminate('x'), 1), {3, 1}}
    };// { 1 + 3(x1) }

    auto hash1 = Hash<Polynomial>{}(p1);
    auto hash2 = Hash<Polynomial>{}(p2);
    auto hash3 = Hash<Polynomial>{}(p3);

    EXPECT_EQ(hash1, hash2);// Hashes should be the same for p1 and p2
    EXPECT_NE(hash1, hash3);// Hashes should be different for p1 and p3
}

TEST(TestPolynomial, TestPolynomialDisplay)
{
    Polynomial p1;

    std::stringstream ss1;
    poly_print(ss1, p1);

    EXPECT_EQ(ss1.str(), "{ }");

    Polynomial p2 { {Monomial(), {1, 1}}};
    std::stringstream ss2;
    poly_print(ss2, p2);

    EXPECT_EQ(ss2.str(), "{ 1 }");

    Polynomial p3 { {Monomial(), {1, 1}}, {Monomial(Indeterminate('x', 2), 2), {1, 2}}};
    std::stringstream ss3;
    poly_print(ss3, p3);

    EXPECT_EQ(ss3.str(), "{ 1 1/2(x2^2) }");
}

TEST(TestPolynomial, TestPolynomialAdditionCommonTerms)
{
    Polynomial p1{
            {Monomial(), {1, 1}},
            {Monomial(Indeterminate('x'), 1), {2, 1}}
    };// { 1 + 2(x1) }
    Polynomial p2{
            {Monomial(Indeterminate('x'), 1), {1, 1}}
    };// { 1(x1) }

    poly_add_inplace(p1, p2);

    Polynomial expected {
            {Monomial(), {1, 1}},
            {Monomial(Indeterminate('x'), 1), {3, 1}}
    };

    EXPECT_EQ(p1, expected);
}


TEST(TestPolynomial, TestPolynomialSubtractionCommonTerms)
{
    Polynomial p1{
            {Monomial(), {1, 1}},
            {Monomial(Indeterminate('x'), 1), {2, 1}}
    };// { 1 + 2(x1) }
    Polynomial p2{
            {Monomial(Indeterminate('x'), 1), {1, 1}}
    };// { 1(x1) }

    poly_sub_inplace(p1, p2);
    Polynomial expected {
            {Monomial(), {1, 1}},
            {Monomial(Indeterminate('x'), 1), {1, 1}}
    }; // { 1 1(x1) }

    EXPECT_EQ(p1, expected);

}

TEST(TestPolynomial, TestPolynomialMultiplication)
{
    Polynomial p1{
            {Monomial(), {1, 1}},
            {Monomial('x', 1), {2, 1}}
    };// { 1 2(x1) }
    Polynomial p2{
            {Monomial('x', 1), {3, 1}}
    };// { 3(x1) }

    Polynomial result(p1);


    poly_mul_inplace(result, p2);
    Polynomial expected {
            {Monomial('x', 1), {3, 1}},
            {Monomial('x', 1, 2), {6, 1}}
    }; // { 3(x1) 6(x1^2) }

    EXPECT_EQ(result, expected) << p1 << ' ' << p2;
}


TEST(TestPolynomial, TestDivisonByRational)
{
    Polynomial p1{
            {Monomial(), {1, 1}},
            {Monomial(Indeterminate('x'), 1), {2, 1}}
    };// { 1 2(x1) }

    generics::dtl::RationalCoeff r(1, 2); // 1/2
    poly_div_inplace(p1, r);

    Polynomial expected {
            {Monomial(), {2, 1}},
            {Monomial(Indeterminate('x'), 1), {4, 1}}
    }; // { 2 2(x1) }

    EXPECT_EQ(p1, expected);
}
