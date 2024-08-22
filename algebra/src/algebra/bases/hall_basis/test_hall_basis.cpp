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

TEST_F(HallBasisTests, CheckKeyTypeConversionDegree3)
{
    const auto word_type = basis->supported_key_types()[0];
    const auto index_type = basis->supported_key_types()[1];

    rpy::dimn_t indices[] = {
            6,// [1,[1,2]]
            9,
    };

    for (auto& index : indices) {
        auto key = basis->to_key(index);
        auto roundtrip_index = basis->to_index(key);
        EXPECT_EQ(index, roundtrip_index) << key;
    }
}

TEST_F(HallBasisTests, CheckDegreeFunctionLetters)
{
    const auto word_type = basis->supported_key_types()[0];
    const auto index_type = basis->supported_key_types()[1];

    // degree 1
    const rpy::dimn_t letters[] = {0, 1, 2};
    for (auto& index : letters) {
        auto key = basis->to_key(index);
        EXPECT_EQ(basis->degree(key), 1);
        EXPECT_EQ(basis->degree({&index, index_type}), 1);
    }
}

TEST_F(HallBasisTests, CheckDegreeFunctionPairs)
{
    const auto word_type = basis->supported_key_types()[0];
    const auto index_type = basis->supported_key_types()[1];

    const rpy::dimn_t pairs[] = {3, 4, 5};
    for (auto& index : pairs) {
        auto key = basis->to_key(index);
        EXPECT_EQ(basis->degree(key), 2);
        EXPECT_EQ(basis->degree({&index, index_type}), 2);
    }
}

TEST_F(HallBasisTests, CheckKeyEquals)
{
    const auto word_type = basis->supported_key_types()[0];
    const auto index_type = basis->supported_key_types()[1];

    const rpy::dimn_t indices[] = {0, 1, 2, 5, 10, 27, 40, 77};

    for (auto& index : indices) {
        auto key = basis->to_key(index);
        EXPECT_TRUE(basis->equals(key, {&index, index_type}));
    }
}

TEST_F(HallBasisTests, CheckKeyHashEqual)
{
    const auto word_type = basis->supported_key_types()[0];
    const auto index_type = basis->supported_key_types()[1];

    const rpy::dimn_t indices[] = {0, 1, 2, 5, 10, 27, 40, 77};

    for (auto& index : indices) {
        auto key = basis->to_key(index);
        EXPECT_EQ(basis->hash(key), basis->hash({&index, index_type}));
    }
}

TEST_F(HallBasisTests, CheckParentsWordLetter)
{
    const rpy::dimn_t letters[] = {0, 1, 2};

    for (auto& index : letters) {
        auto key = basis->to_key(index);
        auto parents = basis->parents(key);

        EXPECT_FALSE(parents.first.fast_is_zero());
        EXPECT_EQ(parents.second, key);
    }
}

TEST_F(HallBasisTests, CheckParentsWordPair)
{
    rpy::dimn_t index = 3;// [1,2]
    auto key = basis->to_key(index);
    auto parents = basis->parents(key);

    EXPECT_EQ(parents.first, basis->to_key(0));
    EXPECT_EQ(parents.second, basis->to_key(1));
}

TEST_F(HallBasisTests, CheckParentsIndexLetter)
{
    const auto index_type = basis->supported_key_types()[1];
    const rpy::dimn_t letters[] = {0, 1, 2};

    for (auto& index : letters) {
        BasisKeyCRef key{&index, index_type};
        auto parents = basis->parents(key);

        EXPECT_FALSE(parents.first.fast_is_zero());
        EXPECT_EQ(parents.second, key);
    }
}

TEST_F(HallBasisTests, CheckParentsIndexPair)
{
    const auto index_type = basis->supported_key_types()[1];

    rpy::dimn_t index = 3;// [1,2]
    BasisKeyCRef key{&index, index_type};
    auto parents = basis->parents(key);

    rpy::dimn_t left_idx = 0;
    BasisKeyCRef left{&left_idx, index_type};
    EXPECT_EQ(parents.first, left);

    rpy::dimn_t right_idx = 11;
    BasisKeyCRef right{&right_idx, index_type};
    EXPECT_EQ(parents.second, right);
}

TEST_F(HallBasisTests, CheckIsLetter)
{
    const auto index_type = basis->supported_key_types()[1];
    const rpy::dimn_t letters[] = {0, 1, 2};

    for (auto& index : letters) {
        BasisKeyCRef key{&index, index_type};

        EXPECT_TRUE(basis->is_letter(key));
    }

    rpy::dimn_t words[] = {4, 7, 15, 22, 55, 72};
    for (auto& index : words) {
        EXPECT_FALSE(basis->is_letter({&index, index_type}));
    }
}


TEST_F(HallBasisTests, ChecKToStringWordLetter)
{
    auto key = basis->to_key(0);

    EXPECT_EQ(basis->to_string(key), "1");
}

TEST_F(HallBasisTests, ChecKToStringWordPair)
{
    auto key = basis->to_key(3);

    EXPECT_EQ(basis->to_string(key), "[1,2]");
}

TEST_F(HallBasisTests, CheckToStringWordHigher)
{
    auto key = basis->to_key(6); // [1,[1,2]]

    EXPECT_EQ(basis->to_string(key), "[1,[1,2]]");
}


TEST_F(HallBasisTests, ChecKToStringIndexLetter)
{
    BasisKey key(basis->supported_key_types()[1], 0);

    EXPECT_EQ(basis->to_string(key), "1");
}

TEST_F(HallBasisTests, ChecKToStringIndexPair)
{
    BasisKey key(basis->supported_key_types()[1], 0); // [1,2]

    EXPECT_EQ(basis->to_string(key), "[1,2]");
}

TEST_F(HallBasisTests, CheckToStringIndexHigher)
{
    BasisKey key(basis->supported_key_types()[1], 0);// [1,[1,2]]

    EXPECT_EQ(basis->to_string(key), "[1,[1,2]]");
}
