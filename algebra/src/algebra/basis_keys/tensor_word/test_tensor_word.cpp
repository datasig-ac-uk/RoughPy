//
// Created by sam on 8/15/24.
//

// ReSharper disable CppLocalVariableMayBeConst
#include <gtest/gtest.h>

#include "tensor_word.h"
#include <roughpy/core/hash.h>

#include <sstream>

namespace rpy {
namespace algebra {

void PrintTo(const TensorWord& word, std::ostream* os) { word.print(*os); }

}// namespace algebra
}// namespace rpy

using namespace rpy::algebra;
using rpy::let_t;

TEST(TensorWordTests, TestDegreeOfEmptyWord)
{
    TensorWord emptyword;
    EXPECT_EQ(emptyword.degree(), 0);
}

TEST(TensorWordTests, TestDegreeOfLetter)
{
    TensorWord word{let_t(1)};
    EXPECT_EQ(word.degree(), 1);
}

TEST(TensorWordTests, TestDegreeOfPair)
{
    TensorWord word{1, 2};
    EXPECT_EQ(word.degree(), 2);
}

TEST(TensorWordTests, TestDegreeHigher)
{
    TensorWord word{1, 1, 1, 1, 1};
    EXPECT_EQ(word.degree(), 5);
}

TEST(TensorWordTests, TestLeftParentOnEmptyWord)
{
    TensorWord emptyword;
    EXPECT_EQ(emptyword.left_parent(), emptyword);
}

TEST(TensorWordTests, TestLeftParentOnSingleLetter)
{
    TensorWord word{let_t(1)};
    TensorWord expected{};
    EXPECT_EQ(word.left_parent(), expected);
}

TEST(TensorWordTests, TestLeftParentOnPair)
{
    TensorWord word{1, 2};
    TensorWord expected{1};
    EXPECT_EQ(word.left_parent(), expected);
}

TEST(TensorWordTests, TestLeftParentOnMultipleLetters)
{
    TensorWord word{1, 2, 3, 4, 5};
    TensorWord expected{1};
    EXPECT_EQ(word.left_parent(), expected);
}

TEST(TensorWordTests, TestRightParentOnEmptyWord)
{
    TensorWord emptyword;
    EXPECT_EQ(emptyword.right_parent(), emptyword);
}

TEST(TensorWordTests, TestRightParentOnSingleLetter)
{
    TensorWord word{let_t(1)};
    TensorWord expected {1};
    EXPECT_EQ(word.right_parent(), expected);
}

TEST(TensorWordTests, TestRightParentOnPair)
{
    TensorWord word{1, 2};
    TensorWord expected{2};
    EXPECT_EQ(word.right_parent(), expected);
}

TEST(TensorWordTests, TestRightParentOnMultipleLetters)
{
    TensorWord word{1, 2, 3, 4, 5};
    // Assuming right_parent removes the first letter
    TensorWord expected{2, 3, 4, 5};
    EXPECT_EQ(word.right_parent(), expected);
}

TEST(TensorWordTests, TestIsLetterEmptyWord)
{
    TensorWord emptyword;
    EXPECT_FALSE(emptyword.is_letter());
}

TEST(TensorWordTests, TestIsLetterLetter)
{
    TensorWord word{1};
    EXPECT_TRUE(word.is_letter());
}

TEST(TensorWordTests, TestIsLetterHigher)
{
    TensorWord word{1, 1, 1, 1, 1};
    EXPECT_FALSE(word.is_letter());
}

TEST(TensorWordTests, TestToStringEmptyWord)
{
    TensorWord emptyword;

    std::stringstream ss;
    emptyword.print(ss);
    EXPECT_EQ(ss.str(), "");
}

TEST(TensorWordTests, TestToStringSingleLetter)
{
    TensorWord word{let_t(1)};
    std::stringstream ss;
    word.print(ss);
    EXPECT_EQ(ss.str(), "1");
}
TEST(TensorWordTests, TestToStringPair)
{
    TensorWord word{1, 2};
    std::stringstream ss;
    word.print(ss);
    EXPECT_EQ(ss.str(), "1,2");
}
TEST(TensorWordTests, TestToStringMultipleLetters)
{
    TensorWord word{1, 2, 3, 4, 5};
    std::stringstream ss;
    word.print(ss);
    EXPECT_EQ(ss.str(), "1,2,3,4,5");
}
TEST(TensorWordTests, TestToStringWithRepeatingLetters)
{
    TensorWord word{1, 1, 2, 2, 3};
    std::stringstream ss;
    word.print(ss);
    EXPECT_EQ(ss.str(), "1,1,2,2,3");
}

TEST(TensorWordTests, TestMinAlphabetSizeEmptyWord)
{
    TensorWord emptyword;
    EXPECT_EQ(emptyword.min_alphabet_size(), 0);
}

TEST(TensorWordTests, TestMinAlphabetSizeSingleLetter)
{
    TensorWord word{let_t(1)};
    EXPECT_EQ(word.min_alphabet_size(), 1);
}

TEST(TensorWordTests, TestMinAlphabetSizeMultipleUniqueLetters)
{
    TensorWord word{1, 2, 3};
    EXPECT_EQ(word.min_alphabet_size(), 3);
}

TEST(TensorWordTests, TestMinAlphabetSizeRepeatingLetters)
{
    TensorWord word{1, 2, 2, 3, 3, 3};
    EXPECT_EQ(word.min_alphabet_size(), 3);
}

TEST(TensorWordTests, TestMinAlphabetSizeAllSameLetter)
{
    TensorWord word{1, 1, 1, 1};
    EXPECT_EQ(word.min_alphabet_size(), 1);
}

TEST(TensorWordTests, TestEqualityOperatorEmptyWords)
{
    TensorWord word1;
    TensorWord word2;
    EXPECT_TRUE(word1 == word2);
}

TEST(TensorWordTests, TestEqualityOperatorSingleLetter)
{
    TensorWord word1{let_t(1)};
    TensorWord word2{let_t(1)};
    EXPECT_TRUE(word1 == word2);
}

TEST(TensorWordTests, TestEqualityOperatorDifferentSingleLetters)
{
    TensorWord word1{let_t(1)};
    TensorWord word2{let_t(2)};
    EXPECT_FALSE(word1 == word2);
}

TEST(TensorWordTests, TestEqualityOperatorMultipleLettersSameOrder)
{
    TensorWord word1{1, 2, 3};
    TensorWord word2{1, 2, 3};
    EXPECT_TRUE(word1 == word2);
}

TEST(TensorWordTests, TestEqualityOperatorMultipleLettersDifferentOrder)
{
    TensorWord word1{1, 2, 3};
    TensorWord word2{3, 2, 1};
    EXPECT_FALSE(word1 == word2);
}

TEST(TensorWordTests, TestEqualityOperatorDifferentSizes)
{
    TensorWord word1{1, 2, 3};
    TensorWord word2{1, 2};
    EXPECT_FALSE(word1 == word2);
}

TEST(TensorWordTests, TestHashValueEmptyWord)
{
    TensorWord word;
    auto hash_value = rpy::Hash<TensorWord>()(word);
    EXPECT_EQ(hash_value, rpy::Hash<TensorWord>()(word));
}

TEST(TensorWordTests, TestHashValueSingleLetterWord)
{
    TensorWord word{1};
    auto hash_value = rpy::Hash<TensorWord>()(word);
    EXPECT_EQ(hash_value, rpy::Hash<TensorWord>()(word));
}

TEST(TensorWordTests, TestHashValueSameLetterWords)
{
    TensorWord word1{1};
    TensorWord word2{1};
    auto hash_value1 = rpy::Hash<TensorWord>()(word1);
    auto hash_value2 = rpy::Hash<TensorWord>()(word2);
    EXPECT_EQ(hash_value1, hash_value2);
}

TEST(TensorWordTests, TestHashValueDifferentLetterWords)
{
    TensorWord word1{1};
    TensorWord word2{2};
    auto hash_value1 = rpy::Hash<TensorWord>()(word1);
    auto hash_value2 = rpy::Hash<TensorWord>()(word2);
    EXPECT_NE(hash_value1, hash_value2);
}

TEST(TensorWordTests, TestHashValueMultipleLettersSameOrder)
{
    TensorWord word1{1, 2, 3};
    TensorWord word2{1, 2, 3};
    auto hash_value1 = rpy::Hash<TensorWord>()(word1);
    auto hash_value2 = rpy::Hash<TensorWord>()(word2);
    EXPECT_EQ(hash_value1, hash_value2);
}

TEST(TensorWordTests, TestHashValueMultipleLettersDifferentOrder)
{
    TensorWord word1{1, 2, 3};
    TensorWord word2{3, 2, 1};
    auto hash_value1 = rpy::Hash<TensorWord>()(word1);
    auto hash_value2 = rpy::Hash<TensorWord>()(word2);
    EXPECT_NE(hash_value1, hash_value2);
}

TEST(TensorWordTests, TestHashValueDifferentSizes)
{
    TensorWord word1{1, 2, 3};
    TensorWord word2{1, 2};
    auto hash_value1 = rpy::Hash<TensorWord>()(word1);
    auto hash_value2 = rpy::Hash<TensorWord>()(word2);
    EXPECT_NE(hash_value1, hash_value2);
}

// Test case for less than comparison
TEST(TensorWordTests, LessThanComparisons)
{
    TensorWord word1{1, 2, 3};
    TensorWord word2{1, 2, 3, 4};
    TensorWord word3{1, 2, 4};
    TensorWord word4{1, 2, 3};// same as word1

    EXPECT_TRUE(word1 < word2); // word1 has less degree than word2
    EXPECT_TRUE(word1 < word3); // word1 precedes word3 lexicographically
    EXPECT_FALSE(word1 < word4);// word1 is equal to word4
    EXPECT_FALSE(word2 < word1);// word2 has higher degree than word1
}

// Test case for not less than comparison
TEST(TensorWordTests, NotLessThanComparisons)
{
    TensorWord word1{1, 2, 3};
    TensorWord word2{1, 2, 3, 4};
    TensorWord word3{1, 2, 4};
    TensorWord word4{1, 2, 3};// same as word1

    EXPECT_FALSE(word2 < word1);// word2 has higher degree than word1
    EXPECT_FALSE(word3 < word1);// word1 precedes word3 lexicographically
    EXPECT_FALSE(word4 < word1);// word1 is equal to word4
}

// Test case for equals comparison
TEST(TensorWordTests, EqualsComparisons)
{
    TensorWord word1{1, 2, 3};
    TensorWord word2{1, 2, 3, 4};
    TensorWord word3{1, 2, 4};
    TensorWord word4{1, 2, 3};// same as word1

    EXPECT_TRUE(word1 == word4); // word1 is equal to word4
    EXPECT_FALSE(word1 == word2);// word1 is not equal to word2
    EXPECT_FALSE(word1 == word3);// word1 is not equal to word3
}

// Test case for not equals comparison
TEST(TensorWordTests, NotEqualsComparisons)
{
    TensorWord word1{1, 2, 3};
    TensorWord word2{1, 2, 3, 4};
    TensorWord word3{1, 2, 4};
    TensorWord word4{1, 2, 3};// same as word1

    EXPECT_TRUE(word1 != word2); // word1 is not equal to word2
    EXPECT_TRUE(word1 != word3); // word1 is not equal to word3
    EXPECT_FALSE(word1 != word4);// word1 is equal to word4
}
