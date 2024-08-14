//
// Created by sam on 14/08/24.
//

#include <gtest/gtest.h>

#include "lie_word.h"

namespace rpy {
namespace algebra {

void PrintTo(const LieWord& word, std::ostream* os) { *os << word; }

}// namespace algebra
}// namespace rpy

using namespace rpy::algebra;
using rpy::let_t;

TEST(LieWordTests, TestDegreeOfLetter)
{
    LieWord word{let_t(1)};
    EXPECT_EQ(word.degree(), 1);
}

TEST(LieWordTests, TestDegreeOfPair)
{
    LieWord word{let_t(1), let_t(2)};
    EXPECT_EQ(word.degree(), 2);
}

TEST(LieWordTests, TestDegreeOfWordCombination)
{
    LieWord left{let_t(1)}, right{let_t(1), let_t(2)};
    LieWord word{left, right};

    EXPECT_EQ(word.degree(), left.degree() + right.degree());
}

TEST(LieWordTests, TestToStringLetter)
{
    LieWord word{let_t(1)};
    std::stringstream ss;
    word.print(ss);
    EXPECT_EQ(ss.str(), "1");
}

TEST(LieWordTests, TestToStringPair)
{
    LieWord word{let_t(1), let_t(2)};
    std::stringstream ss;
    word.print(ss);
    EXPECT_EQ(ss.str(), "[1,2]");
}

TEST(LieWordTests, TestToStringCompound)
{
    LieWord left{let_t(1)}, right{let_t(1), let_t(2)};
    LieWord word{left, right};

    std::stringstream ss;
    word.print(ss);
    EXPECT_EQ(ss.str(), "[1,[1,2]]");
}

TEST(LieWordTests, TestMinAlphabetSizeSingleLetter)
{
    LieWord word{let_t(1)};
    EXPECT_EQ(word.min_alphabet_size(), 1);
}

TEST(LieWordTests, TestMinAlphabetSizePair)
{
    LieWord word{let_t(1), let_t(2)};
    EXPECT_EQ(word.min_alphabet_size(), 2);
}

TEST(LieWordTests, TestMinAlphabetSizeWordCombination)
{
    LieWord left{let_t(1)}, right{let_t(1), let_t(2)};
    LieWord word{left, right};
    EXPECT_EQ(word.min_alphabet_size(), 2);
}

TEST(LieWordTests, TestMinAlphabetSizeWithRepeatedLetters)
{
    LieWord word{
            let_t(1),
            LieWord{let_t(2), let_t(1)}
    };
    EXPECT_EQ(word.min_alphabet_size(), 2);
}

TEST(LieWordTests, TestIsLetterSingleLetter)
{
    LieWord word{let_t(1)};
    EXPECT_TRUE(word.is_letter());
}

TEST(LieWordTests, TestIsLetterPair)
{
    LieWord word{let_t(1), let_t(2)};
    EXPECT_FALSE(word.is_letter());
}

TEST(LieWordTests, TestIsLetterCompound)
{
    LieWord left{let_t(1)}, right{let_t(1), let_t(2)};
    LieWord word{left, right};
    EXPECT_FALSE(word.is_letter());
}

TEST(LieWordTests, TestEqualitySingleLetters)
{
    LieWord word1{let_t(1)};
    LieWord word2{let_t(1)};
    LieWord word3{let_t(2)};

    EXPECT_EQ(word1, word2);
    EXPECT_NE(word1, word3);
}

TEST(LieWordTests, TestEqualityPairs)
{
    LieWord word1{let_t(1), let_t(2)};
    LieWord word2{let_t(1), let_t(2)};
    LieWord word3{let_t(2), let_t(1)};

    EXPECT_EQ(word1, word2);
    EXPECT_NE(word1, word3);
}

TEST(LieWordTests, TestEqualityCompounds)
{
    LieWord left1{let_t(1)};
    LieWord right1{let_t(1), let_t(2)};
    LieWord word1{left1, right1};

    LieWord left2{let_t(1)};
    LieWord right2{let_t(1), let_t(2)};
    LieWord word2{left2, right2};

    LieWord left3{let_t(2)};
    LieWord right3{let_t(1), let_t(2)};
    LieWord word3{left3, right3};

    EXPECT_EQ(word1, word2);
    EXPECT_NE(word1, word3);
}

TEST(LieWordTests, TestEqualityNestedCompounds)
{
    LieWord inner1{let_t(1), let_t(2)};
    LieWord outer1{inner1, let_t(3)};
    LieWord word1{outer1, let_t(4)};

    LieWord inner2{let_t(1), let_t(2)};
    LieWord outer2{inner2, let_t(3)};
    LieWord word2{outer2, let_t(4)};

    LieWord inner3{let_t(1), let_t(2)};
    LieWord outer3{inner3, let_t(3)};
    LieWord word3{outer3, let_t(5)};

    EXPECT_EQ(word1, word2);
    EXPECT_NE(word1, word3);
}

TEST(LieWordTests, TestParentsLetter)
{
    LieWord word{let_t(1)};

    auto left = word.left_parent();
    auto right = word.right_parent();

    ASSERT_TRUE(left);
    EXPECT_FALSE(right);
    EXPECT_EQ(*left, word);
}

TEST(LieWordTests, TestParentsPair)
{
    LieWord left_word{let_t(1)}, right_word{let_t(2)};
    LieWord word{left_word, right_word};

    auto left = word.left_parent();
    auto right = word.right_parent();
    ASSERT_TRUE(left);
    ASSERT_TRUE(right);

    EXPECT_EQ(*left, left_word);
    EXPECT_EQ(*right, right_word);
}

TEST(LieWordTests, TestParentsCompound)
{
    LieWord left_word{let_t(1), let_t(2)};
    LieWord right_word{let_t(1)};
    LieWord word{left_word, right_word};

    auto left = word.left_parent();
    auto right = word.right_parent();

    ASSERT_TRUE(left);
    ASSERT_TRUE(right);

    EXPECT_EQ(*left, left_word);
    EXPECT_EQ(*right, right_word);
}

TEST(LieWordTests, TestHashValueSingleLetter)
{
    LieWord word{let_t(1)};
    rpy::Hash<LieWord> hash_fn;
    auto hash_value1 = hash_fn(word);

    LieWord same_word{let_t(1)};
    auto hash_value2 = hash_fn(same_word);

    EXPECT_EQ(hash_value1, hash_value2);
}

TEST(LieWordTests, TestHashValueDifferentLetters)
{
    LieWord word1{let_t(1)};
    LieWord word2{let_t(2)};
    rpy::Hash<LieWord> hash_fn;

    auto hash_value1 = hash_fn(word1);
    auto hash_value2 = hash_fn(word2);

    EXPECT_NE(hash_value1, hash_value2);
}

TEST(LieWordTests, TestHashValueSamePairs)
{
    LieWord word1{let_t(1), let_t(2)};
    LieWord word2{let_t(1), let_t(2)};
    rpy::Hash<LieWord> hash_fn;

    auto hash_value1 = hash_fn(word1);
    auto hash_value2 = hash_fn(word2);

    EXPECT_EQ(hash_value1, hash_value2);
}

TEST(LieWordTests, TestHashValueDifferentPairs)
{
    LieWord word1{let_t(1), let_t(2)};
    LieWord word2{let_t(2), let_t(1)};
    rpy::Hash<LieWord> hash_fn;

    auto hash_value1 = hash_fn(word1);
    auto hash_value2 = hash_fn(word2);

    EXPECT_NE(hash_value1, hash_value2);
}

TEST(LieWordTests, TestHashValueSameCompoundWords)
{
    LieWord left1{let_t(1)};
    LieWord right1{let_t(1), let_t(2)};
    LieWord word1{left1, right1};

    LieWord left2{let_t(1)};
    LieWord right2{let_t(1), let_t(2)};
    LieWord word2{left2, right2};

    rpy::Hash<LieWord> hash_fn;

    auto hash_value1 = hash_fn(word1);
    auto hash_value2 = hash_fn(word2);

    EXPECT_EQ(hash_value1, hash_value2);
}

TEST(LieWordTests, TestHashValueDifferentCompoundWords)
{
    LieWord left1{let_t(1)};
    LieWord right1{let_t(1), let_t(2)};
    LieWord word1{left1, right1};

    LieWord left2{let_t(2)};
    LieWord right2{let_t(1), let_t(2)};
    LieWord word2{left2, right2};

    rpy::Hash<LieWord> hash_fn;

    auto hash_value1 = hash_fn(word1);
    auto hash_value2 = hash_fn(word2);

    EXPECT_NE(hash_value1, hash_value2);
}