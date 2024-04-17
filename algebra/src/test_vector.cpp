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
