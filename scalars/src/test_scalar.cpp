// Copyright (c) 2023 RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

//
// Created by sam on 13/03/23.
//

#include <gtest/gtest.h>

#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_type.h>

#include <roughpy/platform/serialization.h>
#include <sstream>

using namespace rpy;
using namespace rpy::scalars;

TEST(scalar, CreateFromDoubleTypeInference)
{
    Scalar scalar(3.142);

    ASSERT_EQ(scalar.type(), ScalarType::of<double>());
}

TEST(scalar, CreateWithTypeCoercion)
{
    Scalar scalar(ScalarType::of<float>(), 3.142);

    ASSERT_EQ(scalar.type(), ScalarType::of<float>());
    ASSERT_FLOAT_EQ(scalar_cast<float>(scalar), 3.142f);
}

TEST(scalar, CreateWithNumAndDenomTypeGiven)
{
    Scalar scalar(ScalarType::of<float>(), 1, 8);

    ASSERT_EQ(scalar_cast<float>(scalar), 1.0f / 8.0f);
}

TEST(scalar, DefaultCreateAssignDouble)
{
    Scalar scalar;

    EXPECT_EQ(scalar.type(), nullptr);

    scalar = 3.142;

    ASSERT_EQ(scalar.type(), ScalarType::of<double>());
    ASSERT_EQ(scalar_cast<double>(scalar), 3.142);
}

TEST(scalar, DefaultCreateAssignFloat)
{
    Scalar scalar;

    EXPECT_EQ(scalar.type(), nullptr);

    scalar = 3.142f;

    ASSERT_EQ(scalar.type(), ScalarType::of<float>());
    ASSERT_EQ(scalar_cast<float>(scalar), 3.142f);
}

TEST(scalar, AssignDoubleToFloat)
{
    Scalar scalar(ScalarType::of<float>(), 3.142f);

    scalar = 2.718;

    ASSERT_EQ(scalar.type(), ScalarType::of<float>());
    ASSERT_FLOAT_EQ(scalar_cast<float>(scalar), 2.718f);
}

TEST(scalar, IsZeroTrueForDefault)
{
    Scalar scalar;

    ASSERT_TRUE(scalar.is_zero());
}

TEST(scalar, IsZeroTrueForDoubleZero)
{
    Scalar scalar(0.0);

    ASSERT_TRUE(scalar.is_zero());
}

TEST(scalar, IsZeroFalseForDoubleOne)
{
    Scalar scalar(1.0);

    ASSERT_FALSE(scalar.is_zero());
}

TEST(scalar, ToScalarTFromFloat)
{
    Scalar scalar(ScalarType::of<float>(), 3.142f);

    ASSERT_FLOAT_EQ(scalar.to_scalar_t(), 3.142f);
}

TEST(scalar, AdditionSameType)
{
    Scalar lhs(ScalarType::of<float>(), 3.142f);
    Scalar rhs(ScalarType::of<float>(), 2.718f);

    auto result = lhs + rhs;

    ASSERT_FLOAT_EQ(scalar_cast<float>(result), 3.142f + 2.718f);
}

TEST(scalar, SubtractionSameType)
{
    Scalar lhs(ScalarType::of<float>(), 3.142f);
    Scalar rhs(ScalarType::of<float>(), 2.718f);

    auto result = lhs - rhs;

    ASSERT_FLOAT_EQ(scalar_cast<float>(result), 3.142f - 2.718f);
}

TEST(scalar, MultiplicationSameType)
{
    Scalar lhs(ScalarType::of<float>(), 3.142f);
    Scalar rhs(ScalarType::of<float>(), 2.718f);

    auto result = lhs * rhs;

    ASSERT_FLOAT_EQ(scalar_cast<float>(result), 3.142f * 2.718f);
}

TEST(scalar, DivisionSameType)
{
    Scalar lhs(ScalarType::of<float>(), 3.142);
    Scalar rhs(ScalarType::of<float>(), 2.718f);

    auto result = lhs / rhs;

    ASSERT_FLOAT_EQ(scalar_cast<float>(result), 3.142f / 2.718f);
}

TEST(scalar, DivisionByDefaultThrows)
{
    Scalar lhs(ScalarType::of<float>(), 3.142f);
    Scalar rhs;

    ASSERT_THROW(lhs / rhs, std::runtime_error);
}

TEST(scalar, DivisionByZeroSameTypeThrows)
{
    Scalar lhs(ScalarType::of<float>(), 3.142f);
    Scalar rhs(ScalarType::of<float>(), 0.0f);

    ASSERT_THROW(lhs / rhs, std::runtime_error);
}

TEST(scalar, InplaceAdditionSameType)
{
    Scalar lhs(ScalarType::of<float>(), 3.142f);
    Scalar rhs(ScalarType::of<float>(), 2.718f);

    lhs += rhs;

    ASSERT_FLOAT_EQ(scalar_cast<float>(lhs), 3.142f + 2.718f);
}

TEST(scalar, InplaceSubtractionSameType)
{
    Scalar lhs(ScalarType::of<float>(), 3.142f);
    Scalar rhs(ScalarType::of<float>(), 2.718f);

    lhs -= rhs;

    ASSERT_FLOAT_EQ(scalar_cast<float>(lhs), 3.142f - 2.718f);
}

TEST(scalar, InplaceMultiplicationSameType)
{
    Scalar lhs(ScalarType::of<float>(), 3.142f);
    Scalar rhs(ScalarType::of<float>(), 2.718f);

    lhs *= rhs;

    ASSERT_FLOAT_EQ(scalar_cast<float>(lhs), 3.142f * 2.718f);
}

TEST(scalar, InplaceDivisionByDefaultThrows)
{
    Scalar lhs(ScalarType::of<float>(), 3.142);
    Scalar rhs;

    ASSERT_THROW(lhs /= rhs, std::runtime_error);
}

TEST(scalar, InplaceDivisionSameTypeZeroThrows)
{
    Scalar lhs(ScalarType::of<float>(), 3.142f);
    Scalar rhs(ScalarType::of<float>(), 0.0f);

    ASSERT_THROW(lhs /= rhs, std::runtime_error);
}

TEST(scalar, InplaceDivisionSameType)
{
    Scalar lhs(ScalarType::of<float>(), 3.142f);
    Scalar rhs(ScalarType::of<float>(), 2.718f);

    lhs /= rhs;

    ASSERT_FLOAT_EQ(scalar_cast<float>(lhs), 3.142f / 2.718f);
}

TEST(scalar, EqualsDefaultDefaultTrue)
{
    Scalar lhs;
    Scalar rhs;

    ASSERT_EQ(lhs, rhs);
}

TEST(scalar, EqualsDefaultFloatZeroTrue)
{
    Scalar lhs(ScalarType::of<float>(), 0.0f);
    Scalar rhs;

    ASSERT_EQ(lhs, rhs);
}

TEST(scalar, EqualsDefaultFloatOneFalse)
{
    Scalar lhs(ScalarType::of<float>(), 1.0f);
    Scalar rhs;

    ASSERT_FALSE(lhs == rhs);
}

TEST(Scalar, SerializeSimpleFloatValue)
{
    Scalar val(ScalarType::of<float>(), 1.0f);
    std::stringstream ss;
    {
        archives::JSONOutputArchive out_ar(ss);
        out_ar(val);
    }

    Scalar inval;
    {
        archives::JSONInputArchive in_ar(ss);
        in_ar(inval);
    }

    ASSERT_EQ(val, inval);
}
