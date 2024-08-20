//
// Created by sam on 20/08/24.
//

#include <gtest/gtest.h>


#include "hall_basis.h"



using namespace rpy::algebra;

namespace {


class HallBasisTests : public ::testing::Test
{
public:
    rpy::Rc <const HallBasis> basis;

    void SetUp() override
    {
        basis = HallBasis::get(3, 5);
    }

};

}

TEST_F(HallBasisTests, TestWidthAndDepth)
{
    EXPECT_EQ(basis->alphabet_size(), 3);
    EXPECT_EQ(basis->max_degree(), 3);
}

TEST_F(HallBasisTests, TestSizeOfHallSet)
{

    EXPECT_EQ(basis->max_dimension(), 3906);
}
