//
// Created by sam on 19/11/24.
//

#include <gtest/gtest.h>


#include "roughpy/platform/alloc.h"

#include <boost/align/detail/is_alignment.hpp>

using namespace rpy;


TEST(TestAlignedAlloc, TestIsAlignmentCorrect)
{
    size_t alignments[] = { 1, 2, 4, 8, 16, 32, 64,
                          128, 256, 512, 1024, 2048, 4096};
    for (size_t align : alignments) {
        EXPECT_TRUE(mem::is_alignment(align));
    }
}

TEST(TestAlignedAlloc, TestIsAlignment0False)
{
    EXPECT_FALSE(mem::is_alignment(0));
}

TEST(TestAlignmentAlloc, TestIsAlignmentNonPow2False)
{
    for (size_t nonalign : { 3, 7, 13, 17, 33}) {
        EXPECT_FALSE(mem::is_alignment(nonalign));
    }
}


TEST(TestAlignedAlloc, TestAllocCorrectAlignment)
{
    void* ptr;

    for (size_t align : { 16, 32, 64, 128}) {
        ptr = mem::aligned_alloc(align, 32);
        ASSERT_NE(ptr, nullptr);

        EXPECT_TRUE(mem::is_pointer_aligned(ptr, align));
        mem::aligned_free(ptr, 32);
    }
}

