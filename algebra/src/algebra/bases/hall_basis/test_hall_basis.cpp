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
    rpy::deg_t width;
    rpy::deg_t depth;
    rpy::Rc<const HallBasis> basis;

    void SetUp() override
    {
        width = 3;
        depth = 5;
        basis = HallBasis::get(width, depth);
    }
};

}// namespace

TEST_F(HallBasisTests, TestWidthAndDepth)
{
    EXPECT_EQ(basis->alphabet_size(), width);
    EXPECT_EQ(basis->max_degree(), depth);
}

TEST_F(HallBasisTests, TestSizeOfHallSet)
{
    EXPECT_EQ(basis->max_dimension(), 80);
}

TEST_F(HallBasisTests, ChecKSupportedKeyTypes)
{
    const auto supported_types = basis->supported_key_types();

    ASSERT_EQ(supported_types.size(), 2);
    EXPECT_EQ(supported_types[0]->id(), "lie_word");
    EXPECT_EQ(supported_types[1]->id(), "index_key");

    EXPECT_TRUE(basis->supports_key_type(supported_types[0]));
    EXPECT_TRUE(basis->supports_key_type(supported_types[1]));
    EXPECT_FALSE(basis->supports_key_type(rpy::devices::get_type<int64_t>()));
}

TEST_F(HallBasisTests, CheckHasKeyIndexKey)
{
    const auto index_type = basis->supported_key_types()[1];

    const rpy::dimn_t indexes[] = {0, 1, 2, 5, 25};
    for (auto& index : indexes) {
        EXPECT_TRUE(basis->has_key({&index, index_type}));
    }

    rpy::dimn_t bad_index = 12876;
    EXPECT_FALSE(basis->has_key({&bad_index, index_type}));
}

TEST_F(HallBasisTests, CheckKeyTypeConversionLetters)
{
    const auto word_type = basis->supported_key_types()[0];
    const auto index_type = basis->supported_key_types()[1];

    const rpy::dimn_t indices[] = {0, 1, 2};
    for (auto& index : indices) {
        auto key = basis->to_key(index);
        auto roundtrip_index = basis->to_index(key);
        EXPECT_EQ(index, roundtrip_index);
    }
}

TEST_F(HallBasisTests, CheckKeyTypeConversionPairs)
{

    const auto word_type = basis->supported_key_types()[0];
    const auto index_type = basis->supported_key_types()[1];

    rpy::dimn_t index = 3;//  [1, 2]
    auto key = basis->to_key(index);
    auto roundtrip_index = basis->to_index(key);
    EXPECT_EQ(index, roundtrip_index) << key;
}
