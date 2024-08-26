//
// Created by sam on 26/08/24.
//

#include <gtest/gtest.h>

#include "roughpy/algebra/algebra_fwd.h"
#include "roughpy/algebra/basis.h"
#include "tensor_basis.h"

using namespace rpy::algebra;

class RPY_LOCAL TensorBasisTests : public testing::Test
{


public:
    rpy::deg_t width;
    rpy::deg_t depth;
    BasisPointer basis;

    void SetUp() override
    {
        width = 3;
        depth = 5;
        basis = TensorBasis::get(width, depth);
    }
};


TEST_F(TensorBasisTests, TestSizeOfDegree)
{
    EXPECT_EQ(basis->degree_to_dimension(0), 1);
    EXPECT_EQ(basis->degree_to_dimension(1), 1 + width);
}