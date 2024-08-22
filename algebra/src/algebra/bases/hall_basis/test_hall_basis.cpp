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
    EXPECT_EQ(basis->degree_to_dimension(1), 3);
    EXPECT_EQ(basis->degree_to_dimension(2), 6);
    EXPECT_EQ(basis->degree_to_dimension(3), 14);
    EXPECT_EQ(basis->degree_to_dimension(4), 32);
    EXPECT_EQ(basis->degree_to_dimension(5), 80);

    EXPECT_EQ(basis->max_dimension(), 80);
}

TEST_F(HallBasisTests, TestDimensionToDegree)
{
    EXPECT_EQ(basis->dimension_to_degree(0), 1);
    EXPECT_EQ(basis->dimension_to_degree(2), 1);
    EXPECT_EQ(basis->dimension_to_degree(3), 2);
    EXPECT_EQ(basis->dimension_to_degree(5), 2);
    EXPECT_EQ(basis->dimension_to_degree(6), 3);
    EXPECT_EQ(basis->dimension_to_degree(13), 3);
    EXPECT_EQ(basis->dimension_to_degree(14), 4);
    EXPECT_EQ(basis->dimension_to_degree(31), 4);
    EXPECT_EQ(basis->dimension_to_degree(32), 5);
    EXPECT_EQ(basis->dimension_to_degree(79), 5);
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

    const rpy::dimn_t indices[] = {
            0, // 1
            1, // 2
            2, // 3
            5, // [2,3]
            10,// [2,[2,3]]
            27,// [3,[3,[1,3]]]
            40,// [2,[2,[2,[2,3]]]]
            77 // [[2,3],[3,[1,2]]]
    };

    for (auto& index : indices) {
        auto key = basis->to_key(index);
        EXPECT_TRUE(basis->equals(key, {&index, index_type}))
                << key << ' ' << basis->to_string({&index, index_type});
    }
}

TEST_F(HallBasisTests, CheckKeyHashEqual)
{
    const auto word_type = basis->supported_key_types()[0];
    const auto index_type = basis->supported_key_types()[1];

    const rpy::dimn_t indices[] = {
            0, // 1
            1, // 2
            2, // 3
            5, // [2,3]
            10,// [2,[2,3]]
            27,// [3,[3,[1,3]]]
            40,// [2,[2,[2,[2,3]]]]
            77 // [[2,3],[3,[1,2]]]
    };

    for (auto& index : indices) {
        auto key = basis->to_key(index);
        EXPECT_EQ(basis->hash(key), basis->hash({&index, index_type}))
                << key << ' ' << basis->to_string({&index, index_type});
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

        EXPECT_TRUE(parents.first.fast_is_zero());
        EXPECT_EQ(parents.second, key);
    }
}

TEST_F(HallBasisTests, CheckParentsIndexPair)
{
    const auto index_type = basis->supported_key_types()[1];

    rpy::dimn_t index = 3;// [1,2]
    BasisKeyCRef key{&index, index_type};
    auto parents = basis->parents(key);

    rpy::dimn_t left_idx = 0;// 1
    BasisKeyCRef left{&left_idx, index_type};
    EXPECT_EQ(parents.first, left);

    rpy::dimn_t right_idx = 1;// 2
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
    auto key = basis->to_key(6);// [1,[1,2]]

    EXPECT_EQ(basis->to_string(key), "[1,[1,2]]");
}

TEST_F(HallBasisTests, ChecKToStringIndexLetter)
{
    BasisKey key(basis->supported_key_types()[1], 0);

    EXPECT_EQ(basis->to_string(key), "1");
}

TEST_F(HallBasisTests, ChecKToStringIndexPair)
{
    BasisKey key(basis->supported_key_types()[1], 3);// [1,2]

    EXPECT_EQ(basis->to_string(key), "[1,2]");
}

TEST_F(HallBasisTests, CheckToStringIndexHigher)
{
    BasisKey key(basis->supported_key_types()[1], 6);// [1,[1,2]]

    EXPECT_EQ(basis->to_string(key), "[1,[1,2]]");
}

// This is the hall set we use in the tests above. The first number is the index
// and the second part is the expanded key form.
// 0  1
// 1  2
// 2  3
// 3  [1,2]
// 4  [1,3]
// 5  [2,3]
// 6  [1,[1,2]]
// 7  [1,[1,3]]
// 8  [2,[1,2]]
// 9  [2,[1,3]]
// 10 [2,[2,3]]
// 11 [3,[1,2]]
// 12 [3,[1,3]]
// 13 [3,[2,3]]
// 14 [1,[1,[1,2]]]
// 15 [1,[1,[1,3]]]
// 16 [2,[1,[1,2]]]
// 17 [2,[1,[1,3]]]
// 18 [2,[2,[1,2]]]
// 19 [2,[2,[1,3]]]
// 20 [2,[2,[2,3]]]
// 21 [3,[1,[1,2]]]
// 22 [3,[1,[1,3]]]
// 23 [3,[2,[1,2]]]
// 24 [3,[2,[1,3]]]
// 25 [3,[2,[2,3]]]
// 26 [3,[3,[1,2]]]
// 27 [3,[3,[1,3]]]
// 28 [3,[3,[2,3]]]
// 29 [[1,2],[1,3]]
// 30 [[1,2],[2,3]]
// 31 [[1,3],[2,3]]
// 32 [1,[1,[1,[1,2]]]]
// 33 [1,[1,[1,[1,3]]]]
// 34 [2,[1,[1,[1,2]]]]
// 35 [2,[1,[1,[1,3]]]]
// 36 [2,[2,[1,[1,2]]]]
// 37 [2,[2,[1,[1,3]]]]
// 38 [2,[2,[2,[1,2]]]]
// 39 [2,[2,[2,[1,3]]]]
// 40 [2,[2,[2,[2,3]]]]
// 41 [3,[1,[1,[1,2]]]]
// 42 [3,[1,[1,[1,3]]]]
// 43 [3,[2,[1,[1,2]]]]
// 44 [3,[2,[1,[1,3]]]]
// 45 [3,[2,[2,[1,2]]]]
// 46 [3,[2,[2,[1,3]]]]
// 47 [3,[2,[2,[2,3]]]]
// 48 [3,[3,[1,[1,2]]]]
// 49 [3,[3,[1,[1,3]]]]
// 50 [3,[3,[2,[1,2]]]]
// 51 [3,[3,[2,[1,3]]]]
// 52 [3,[3,[2,[2,3]]]]
// 53 [3,[3,[3,[1,2]]]]
// 54 [3,[3,[3,[1,3]]]]
// 55 [3,[3,[3,[2,3]]]]
// 56 [[1,2],[1,[1,2]]]
// 57 [[1,2],[1,[1,3]]]
// 58 [[1,2],[2,[1,2]]]
// 59 [[1,2],[2,[1,3]]]
// 60 [[1,2],[2,[2,3]]]
// 61 [[1,2],[3,[1,2]]]
// 62 [[1,2],[3,[1,3]]]
// 63 [[1,2],[3,[2,3]]]
// 64 [[1,3],[1,[1,2]]]
// 65 [[1,3],[1,[1,3]]]
// 66 [[1,3],[2,[1,2]]]
// 67 [[1,3],[2,[1,3]]]
// 68 [[1,3],[2,[2,3]]]
// 69 [[1,3],[3,[1,2]]]
// 70 [[1,3],[3,[1,3]]]
// 71 [[1,3],[3,[2,3]]]
// 72 [[2,3],[1,[1,2]]]
// 73 [[2,3],[1,[1,3]]]
// 74 [[2,3],[2,[1,2]]]
// 75 [[2,3],[2,[1,3]]]
// 76 [[2,3],[2,[2,3]]]
// 77 [[2,3],[3,[1,2]]]
// 78 [[2,3],[3,[1,3]]]
// 79 [[2,3],[3,[2,3]]]
