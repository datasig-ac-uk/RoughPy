//
// Created by sam on 26/08/24.
//

#include <gtest/gtest.h>

#include <roughpy/core/helpers.h>

#include "roughpy/algebra/algebra_fwd.h"
#include "roughpy/algebra/basis.h"
#include "tensor_basis.h"

using namespace rpy::algebra;

class TensorBasisTests : public testing::Test
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
    rpy::dimn_t size = 1;

    for (rpy::deg_t i = 0; i <= depth; ++i) {
        EXPECT_EQ(basis->degree_to_dimension(i), size) << "degree " << i;
        size = 1 + width * size;
    }

    EXPECT_EQ(
            basis->max_dimension(),
            (rpy::const_power(width, depth + 1) - 1) / (width - 1)
    );
}

TEST_F(TensorBasisTests, TestDimensionToDegree)
{
    EXPECT_EQ(basis->dimension_to_degree(0), 0);
    EXPECT_EQ(basis->dimension_to_degree(1), 1);
    EXPECT_EQ(basis->dimension_to_degree(3), 1);
    EXPECT_EQ(basis->dimension_to_degree(4), 2);
    EXPECT_EQ(basis->dimension_to_degree(5), 2);
    EXPECT_EQ(basis->dimension_to_degree(12), 2);
    EXPECT_EQ(basis->dimension_to_degree(13), 3);
    EXPECT_EQ(basis->dimension_to_degree(39), 3);
    EXPECT_EQ(basis->dimension_to_degree(40), 4);
    EXPECT_EQ(basis->dimension_to_degree(120), 4);
    EXPECT_EQ(basis->dimension_to_degree(121), 5);
    EXPECT_EQ(basis->dimension_to_degree(363), 5);
    EXPECT_EQ(basis->dimension_to_degree(364), 6);
}

TEST_F(TensorBasisTests, TestDenseDimensionAdjustment)
{
    EXPECT_EQ(basis->dense_dimension(0), 0);
    EXPECT_EQ(basis->dense_dimension(1), 1);
    EXPECT_EQ(basis->dense_dimension(2), 4);
    EXPECT_EQ(basis->dense_dimension(4), 4);
    EXPECT_EQ(basis->dense_dimension(5), 13);
    EXPECT_EQ(basis->dense_dimension(13), 13);
    EXPECT_EQ(basis->dense_dimension(14), 40);
    EXPECT_EQ(basis->dense_dimension(40), 40);
    EXPECT_EQ(basis->dense_dimension(41), 121);
    EXPECT_EQ(basis->dense_dimension(121), 121);
    EXPECT_EQ(basis->dense_dimension(122), 364);
    EXPECT_EQ(basis->dense_dimension(364), 364);
}

TEST_F(TensorBasisTests, TestCheckSupportedKeyTypes)
{
    const auto supported_types = basis->supported_key_types();

    EXPECT_EQ(supported_types.size(), 2);
    EXPECT_EQ(supported_types[0]->id(), "tensor_word");
    EXPECT_EQ(supported_types[1]->id(), "index_key");

    EXPECT_TRUE(basis->supports_key_type(supported_types[0]));
    EXPECT_TRUE(basis->supports_key_type(supported_types[1]));

    EXPECT_FALSE(basis->supports_key_type(rpy::devices::get_type<int64_t>()));
}

TEST_F(TensorBasisTests, CheckHasKeyIndexKey)
{
    const auto index_type = basis->supported_key_types()[1];

    for (rpy::dimn_t index : {0, 1, 3, 10, 26, 65, 193}) {
        EXPECT_TRUE(basis->has_key({&index, index_type}));
    }
}

TEST_F(TensorBasisTests, CheckKeyTypeConversionEmptyWord)
{
    auto key = basis->to_key(0);
    auto roundtrip_index = basis->to_index(key);

    EXPECT_EQ(0, roundtrip_index) << key;
}

TEST_F(TensorBasisTests, CheckKeyTypeConversionLetters)
{
    for (rpy::dimn_t index : {1, 2, 3}) {
        auto key = basis->to_key(index);
        auto roundtrip_index = basis->to_index(key);

        EXPECT_EQ(index, roundtrip_index) << key;
    }
}

TEST_F(TensorBasisTests, CheckKeyTypeConversionPairs)
{
    for (rpy::dimn_t index : {4, 5, 6, 7, 8, 9, 10, 11}) {
        auto key = basis->to_key(index);
        auto roundtrip_index = basis->to_index(key);

        EXPECT_EQ(index, roundtrip_index) << key;
    }
}

TEST_F(TensorBasisTests, CheckKeyTypeConversionHigherDegree)
{
    for (rpy::dimn_t index : {13, 44, 126, 193}) {
        auto key = basis->to_key(index);
        auto roundtrip_index = basis->to_index(key);

        EXPECT_EQ(index, roundtrip_index) << key;
    }
}

TEST_F(TensorBasisTests, CheckDegreeFunctionEmptyWord)
{
    const auto word_type = basis->supported_key_types()[0];
    const auto index_type = basis->supported_key_types()[1];

    BasisKey word(word_type);
    BasisKey index(index_type, 0);

    EXPECT_EQ(basis->degree(word), 0);
    EXPECT_EQ(basis->degree(index), 0);
}

TEST_F(TensorBasisTests, CheckDegreeFunctionLetter)
{
    const auto index_type = basis->supported_key_types()[1];

    for (rpy::dimn_t index : {1, 2, 3}) {
        auto word = basis->to_key(index);

        EXPECT_EQ(basis->degree(word), 1);
        EXPECT_EQ(basis->degree({&index, index_type}), 1);
    }
}

TEST_F(TensorBasisTests, CheckDegreeFunctionPairs)
{
    const auto index_type = basis->supported_key_types()[1];

    for (rpy::dimn_t index : {4, 8, 12}) {
        auto word = basis->to_key(index);

        EXPECT_EQ(basis->degree(word), 2);
        EXPECT_EQ(basis->degree({&index, index_type}), 2);
    }
}

TEST_F(TensorBasisTests, CheckDegreeFunctionDegree3)
{
    const auto index_type = basis->supported_key_types()[1];

    for (rpy::dimn_t index : {13, 26, 39}) {
        auto word = basis->to_key(index);

        EXPECT_EQ(basis->degree(word), 3);
        EXPECT_EQ(basis->degree({&index, index_type}), 3);
    }
}

TEST_F(TensorBasisTests, CheckKeyEquals)
{
    const auto index_type = basis->supported_key_types()[1];

    constexpr rpy::dimn_t indices[]
            = {0, 1, 2, 3, 4, 12, 15, 25, 76, 101, 134, 224};

    for (auto& index : indices) {
        auto key = basis->to_key(index);
        EXPECT_TRUE(basis->equals(key, key));
        EXPECT_TRUE(basis->equals({&index, index_type}, {&index, index_type}));
        EXPECT_TRUE(basis->equals(key, {&index, index_type}))
                << key << ' ' << basis->to_string({&index, index_type});
    }
}

TEST_F(TensorBasisTests, CheckKeyHashEqual)
{
    const auto index_type = basis->supported_key_types()[1];

    constexpr rpy::dimn_t indices[]
            = {0, 1, 2, 3, 4, 12, 15, 25, 76, 101, 134, 224};

    for (auto& index : indices) {
        auto key = basis->to_key(index);
        EXPECT_EQ(basis->hash(key), basis->hash({&index, index_type}));
    }
}

TEST_F(TensorBasisTests, CheckParentsEmptyWord)
{
    const auto word_type = basis->supported_key_types()[0];
    const auto index_type = basis->supported_key_types()[1];

    BasisKey word(word_type);
    const auto word_parents = basis->parents(word);

    EXPECT_EQ(word_parents.first, word);
    EXPECT_EQ(word_parents.second, word);

    BasisKey index(index_type, 0);
    const auto index_parents = basis->parents(index);

    EXPECT_EQ(index_parents.first, index);
    EXPECT_EQ(index_parents.second, index);
}

TEST_F(TensorBasisTests, CheckParentsLetter)
{
    const auto word_type = basis->supported_key_types()[0];
    const auto index_type = basis->supported_key_types()[1];

    const rpy::dimn_t index = 2;
    auto word = basis->to_key(index);

    const auto word_parents = basis->parents(word);
    EXPECT_EQ(word_parents.first, BasisKey(word_type));
    EXPECT_EQ(word_parents.second, word);

    const auto index_parents = basis->parents({&index, index_type});
    EXPECT_EQ(index_parents.first, BasisKey(index_type, 0));
    EXPECT_EQ(index_parents.second, BasisKey(index_type, index));
}

TEST_F(TensorBasisTests, CheckParentsPair)
{
    const auto word_type = basis->supported_key_types()[0];
    const auto index_type = basis->supported_key_types()[1];

    const rpy::dimn_t index = 5;// (1,2)
    auto word = basis->to_key(index);

    const auto word_parents = basis->parents(word);
    EXPECT_EQ(word_parents.first, BasisKey(word_type, 1));
    EXPECT_EQ(word_parents.second, BasisKey(word_type, 2));

    const auto index_parents = basis->parents({&index, index_type});
    EXPECT_EQ(index_parents.first, BasisKey(index_type, 1));
    EXPECT_EQ(index_parents.second, BasisKey(index_type, 2));
}

TEST_F(TensorBasisTests, CheckParentsDeg3)
{
    const auto word_type = basis->supported_key_types()[0];
    const auto index_type = basis->supported_key_types()[1];

    const rpy::dimn_t index = 16;// (1,2,3)
    auto word = basis->to_key(index);

    const auto word_parents = basis->parents(word);
    EXPECT_EQ(word_parents.first, BasisKey(word_type, 1));
    EXPECT_EQ(word_parents.second, basis->to_key(5));

    const auto index_parents = basis->parents({&index, index_type});
    EXPECT_EQ(index_parents.first, BasisKey(index_type, 1));
    EXPECT_EQ(index_parents.second, BasisKey(index_type, 5));
}

TEST_F(TensorBasisTests, CheckIsLetterEmptyWord)
{

    const auto word_type = basis->supported_key_types()[0];
    const auto index_type = basis->supported_key_types()[1];

    BasisKey word(word_type);
    BasisKey index(index_type, 0);

    EXPECT_FALSE(basis->is_letter(word));
    EXPECT_FALSE(basis->is_letter(index));
}

TEST_F(TensorBasisTests, CheckIsLetterLetters)
{
    const auto word_type = basis->supported_key_types()[0];
    const auto index_type = basis->supported_key_types()[1];

    for (rpy::dimn_t index : {1, 2, 3}) {
        auto key = basis->to_key(index);
        EXPECT_TRUE(basis->is_letter(key));
        EXPECT_TRUE(basis->is_letter({&index, index_type}));
    }
}

TEST_F(TensorBasisTests, CheckIsLetterHigherDegree)
{
    const auto word_type = basis->supported_key_types()[0];
    const auto index_type = basis->supported_key_types()[1];

    for (rpy::dimn_t index : {4, 8, 15, 22, 36, 49, 63, 125, 183, 211}) {
        auto key = basis->to_key(index);
        EXPECT_FALSE(basis->is_letter(key));
        EXPECT_FALSE(basis->is_letter({&index, index_type}));
    }
}

TEST_F(TensorBasisTests, CheckToStringWordEmptyWord)
{
    const auto word_type = basis->supported_key_types()[0];

    BasisKey word(word_type);

    EXPECT_EQ(basis->to_string(word), "");
}

TEST_F(TensorBasisTests, CheckToStringWordLetter)
{
    rpy::dimn_t index = 2;
    auto word = basis->to_key(index);

    EXPECT_EQ(basis->to_string(word), "2");
}

TEST_F(TensorBasisTests, CheckToStringWordPair)
{
    rpy::dimn_t index = 5;
    auto word = basis->to_key(index);

    EXPECT_EQ(basis->to_string(word), "1,2");
}

TEST_F(TensorBasisTests, CheckToStringWordDeg3)
{
    rpy::dimn_t index = 18;
    auto word = basis->to_key(index);

    EXPECT_EQ(basis->to_string(word), "1,2,3");
}

TEST_F(TensorBasisTests, CheckToStringIndexEmptyWord)
{
    const auto index_type = basis->supported_key_types()[1];

    BasisKey index(index_type, 0);

    EXPECT_EQ(basis->to_string(index), "");
}

TEST_F(TensorBasisTests, CheckToStringIndexLetter)
{
    const auto index_type = basis->supported_key_types()[1];
    rpy::dimn_t index = 2;

    EXPECT_EQ(basis->to_string({&index, index_type}), "2");
}

TEST_F(TensorBasisTests, CheckToStringIndexPair)
{
    const auto index_type = basis->supported_key_types()[1];
    rpy::dimn_t index = 5;

    EXPECT_EQ(basis->to_string({&index, index_type}), "1,2");
}

TEST_F(TensorBasisTests, CheckToStringIndexDeg3)
{
    const auto index_type = basis->supported_key_types()[1];
    rpy::dimn_t index = 18;
    EXPECT_EQ(basis->to_string({&index, index_type}), "1,2,3");
}

// The dimension of the tensor algebra for each degree is as follows
// 0     1
// 1     4
// 2    13
// 3    40
// 4   121
// 5   364
