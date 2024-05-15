//
// Created by sam on 4/17/24.
//

#include <gtest/gtest.h>

#include <roughpy/algebra/tensor_basis.h>
#include <roughpy/algebra/vector.h>

using namespace rpy;
using namespace rpy::algebra;

TEST(VectorTests, VectorAddition)
{
    const auto basis = TensorBasis::get(5, 2);
    const auto stype = *scalars::get_type("f64");

    Vector v1(basis, stype, {1., 2., 3., 4., 5., 6.});
    Vector v2(basis, stype, {-2., 2., -2., 2., -2., 2.});

    Vector vresult(basis, stype, {-1., 4., 1., 6., 3., 8.});

    EXPECT_EQ(vresult, v1 + v2);
}
TEST(VectorTests, VectorSubtraction)
{
    const auto basis = TensorBasis::get(5, 2);
    const auto stype = *scalars::get_type("f64");

    Vector v1(basis, stype, {6., 5., 4., 3., 2., 1.});
    Vector v2(basis, stype, {2., 2., 2., 2., 2., 2.});

    Vector vexpected(basis, stype, {4., 3., 2., 1., 0., -1.});

    EXPECT_EQ(vexpected, v1 - v2);
}

TEST(VectorTests, VectorMultiplicationByScalar)
{
    const auto basis = TensorBasis::get(5, 2);
    const auto stype = *scalars::get_type("f64");

    Vector v1(basis, stype, {1., 2., 3., 4., 5., 6.});
    scalars::Scalar scalar(stype, 2.0);

    Vector vexpected(basis, stype, {2., 4., 6., 8., 10., 12.});

    EXPECT_EQ(vexpected, v1 * scalar);
}

TEST(VectorTests, VectorDivisionByScalar)
{
    const auto basis = TensorBasis::get(5, 2);
    const auto stype = *scalars::get_type("f64");

    Vector v1(basis, stype, {2., 4., 6., 8., 10., 12.});
    scalars::Scalar scalar(stype, 2.0);

    Vector vexpected(basis, stype, {1., 2., 3., 4., 5., 6.});

    EXPECT_EQ(vexpected, v1 / scalar);
}

TEST(VectorTests, StreamOutOperator)
{
    const auto basis = TensorBasis::get(5, 2);
    const auto stype = *scalars::get_type("f64");
    Vector v(basis, stype, {2., 4., 6., 8., 10., 12.});
    std::stringstream ss;
    ss << v;
    std::string expectedOutput = "{ 2() 4(1) 6(2) 8(3) 10(4) 12(5) }";
    EXPECT_EQ(expectedOutput, ss.str());
}

TEST(VectorTests, VectorInplaceAddition)
{
    const auto basis = TensorBasis::get(5, 2);
    const auto stype = *scalars::get_type("f64");
    Vector v1(basis, stype, {1., 2., 3., 4., 5., 6.});
    Vector v2(basis, stype, {-2., 2., -2., 2., -2., 2.});

    v1 += v2;
    Vector vexpected(basis, stype, {-1., 4., 1., 6., 3., 8.});

    EXPECT_EQ(vexpected, v1);
}

TEST(VectorTests, VectorInplaceSubtraction)
{
    const auto basis = TensorBasis::get(5, 2);
    const auto stype = *scalars::get_type("f64");
    Vector v1(basis, stype, {6., 5., 4., 3., 2., 1.});
    Vector v2(basis, stype, {2., 2., 2., 2., 2., 2.});

    v1 -= v2;
    Vector vexpected(basis, stype, {4., 3., 2., 1., 0., -1.});

    EXPECT_EQ(vexpected, v1);
}

TEST(VectorTests, VectorInplaceScalarMultiplication)
{
    const auto basis = TensorBasis::get(5, 2);
    const auto stype = *scalars::get_type("f64");
    Vector v1(basis, stype, {1., 2., 3., 4., 5., 6.});
    scalars::Scalar scalar(stype, 2.0);

    v1 *= scalar;
    Vector vexpected(basis, stype, {2., 4., 6., 8., 10., 12.});

    EXPECT_EQ(vexpected, v1);
}

TEST(VectorTests, VectorInplaceScalarDivision)
{
    const auto basis = TensorBasis::get(5, 2);
    const auto stype = *scalars::get_type("f64");
    Vector v1(basis, stype, {2., 4., 6., 8., 10., 12.});
    scalars::Scalar scalar(stype, 2.0);

    v1 /= scalar;
    Vector vexpected(basis, stype, {1., 2., 3., 4., 5., 6.});

    EXPECT_EQ(vexpected, v1);
}

TEST(VectorTests, VectorEquality)
{
    const auto basis = TensorBasis::get(5, 2);
    const auto stype = *scalars::get_type("f64");
    Vector v1(basis, stype, {1., 2., 3., 4., 5., 6.});
    Vector v2(basis, stype, {1., 2., 3., 4., 5., 6.});
    EXPECT_TRUE(v1 == v2);
}

TEST(VectorTests, VectorNonEquality)
{
    const auto basis = TensorBasis::get(5, 2);
    const auto stype = *scalars::get_type("f64");
    Vector v1(basis, stype, {1., 2., 3., 4., 5., 6.});
    Vector v2(basis, stype, {6., 5., 4., 3., 2., 1.});
    EXPECT_TRUE(v1 != v2);
}

TEST(VectorTests, VectorEqualityWithDifferentSize)
{
    const auto basis = TensorBasis::get(5, 2);
    const auto stype = *scalars::get_type("f64");
    Vector v1(basis, stype, {1., 2., 3., 4., 5., 6., 7.});
    Vector v2(basis, stype, {1., 2., 3., 4., 5., 6.});
    EXPECT_TRUE(v1 != v2);
}

TEST(VectorTests, VectorEqualityWithDifferentSizeZero)
{
    const auto basis = TensorBasis::get(5, 2);
    const auto stype = *scalars::get_type("f64");
    Vector v1(basis, stype, {1., 2., 3., 4., 5., 6., 0.});
    Vector v2(basis, stype, {1., 2., 3., 4., 5., 6.});
    EXPECT_TRUE(v1 == v2);
}

TEST(VectorTests, VectorAdditionRational)
{
    const auto basis = TensorBasis::get(5, 2);
    const auto stype = *scalars::get_type("Rational");

    Vector v1(basis, stype, {1, 2, 3, 4, 5, 6});
    Vector v2(basis, stype, {-2, 2, -2, 2, -2, 2});

    Vector vresult(basis, stype, {-1, 4, 1, 6, 3, 8});

    EXPECT_EQ(vresult, v1 + v2);
}

TEST(VectorTests, VectorSubtractionRational)
{
    const auto basis = TensorBasis::get(5, 2);
    const auto stype = *scalars::get_type("Rational");

    Vector v1(basis, stype, {6, 5, 4, 3, 2, 1});
    Vector v2(basis, stype, {2, 2, 2, 2, 2, 2});

    Vector vexpected(basis, stype, {4, 3, 2, 1, 0, -1});

    EXPECT_EQ(vexpected, v1 - v2);
}

TEST(VectorTests, VectorMultiplicationByScalarRational)
{
    const auto basis = TensorBasis::get(5, 2);
    const auto stype = *scalars::get_type("Rational");

    Vector v1(basis, stype, {1, 2, 3, 4, 5, 6});
    scalars::Scalar scalar(stype, 2);

    Vector vexpected(basis, stype, {2, 4, 6, 8, 10, 12});

    EXPECT_EQ(vexpected, v1 * scalar);
}

TEST(VectorTests, VectorDivisionByScalarRational)
{
    const auto basis = TensorBasis::get(5, 2);
    const auto stype = *scalars::get_type("Rational");

    Vector v1(basis, stype, {2, 4, 6, 8, 10, 12});
    scalars::Scalar scalar(stype, 2);

    Vector vexpected(basis, stype, {1, 2, 3, 4, 5, 6});

    EXPECT_EQ(vexpected, v1 / scalar);
}

TEST(VectorTests, VectorInplaceAdditionRational)
{
    const auto basis = TensorBasis::get(5, 2);
    const auto stype = *scalars::get_type("Rational");

    Vector v1(basis, stype, {1, 2, 3, 4, 5, 6});
    Vector v2(basis, stype, {-2, 2, -2, 2, -2, 2});

    v1 += v2;
    Vector vexpected(basis, stype, {-1, 4, 1, 6, 3, 8});

    EXPECT_EQ(vexpected, v1);
}

TEST(VectorTests, VectorInplaceSubtractionRational)
{
    const auto basis = TensorBasis::get(5, 2);
    const auto stype = *scalars::get_type("Rational");

    Vector v1(basis, stype, {6, 5, 4, 3, 2, 1});
    Vector v2(basis, stype, {2, 2, 2, 2, 2, 2});

    v1 -= v2;
    Vector vexpected(basis, stype, {4, 3, 2, 1, 0, -1});

    EXPECT_EQ(vexpected, v1);
}

TEST(VectorTests, VectorInplaceScalarMultiplicationRational)
{
    const auto basis = TensorBasis::get(5, 2);
    const auto stype = *scalars::get_type("Rational");

    Vector v1(basis, stype, {1, 2, 3, 4, 5, 6});
    scalars::Scalar scalar(stype, 2);

    v1 *= scalar;
    Vector vexpected(basis, stype, {2, 4, 6, 8, 10, 12});

    EXPECT_EQ(vexpected, v1);
}

TEST(VectorTests, VectorInplaceScalarDivisionRational)
{
    const auto basis = TensorBasis::get(5, 2);
    const auto stype = *scalars::get_type("Rational");

    Vector v1(basis, stype, {2, 4, 6, 8, 10, 12});
    scalars::Scalar scalar(stype, 2);

    v1 /= scalar;
    Vector vexpected(basis, stype, {1, 2, 3, 4, 5, 6});

    EXPECT_EQ(vexpected, v1);
}
