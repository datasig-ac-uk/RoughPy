//
// Created by sam on 25/10/24.
//


#include <gtest/gtest.h>


#include <roughpy/platform/memory.h>



using namespace rpy;

namespace {

class AlignedAllocatorTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Code for setting up the test fixture
    }

    void TearDown() override
    {
        // Code for cleaning up after each test
    }
};

TEST_F(AlignedAllocatorTest, AllocationTest)
{
    const size_t alignment = 16;
    const size_t size = 128;
    void* ptr = rpy::align::aligned_alloc(alignment, size);
    ASSERT_NE(ptr, nullptr);
    ASSERT_TRUE(rpy::align::is_pointer_aligned(ptr, alignment));
    align::aligned_free(ptr, size);
}

TEST_F(AlignedAllocatorTest, ZeroAllocationTest)
{
    const size_t alignment = 16;
    const size_t size = 0;
    void* ptr = rpy::align::aligned_alloc(alignment, size);
    ASSERT_EQ(ptr, nullptr);
    align::aligned_free(ptr, size);
}

}// namespace
