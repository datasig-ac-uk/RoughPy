//
// Created by sam on 25/10/24.
//

#include <gtest/gtest.h>

#include <roughpy/platform/memory.h>


using namespace rpy;

TEST(SmalLObjectAllocatorTest, TestUpstreamAllocatorSetCorrectly)
{
    auto* pool_mr = mem::small::get_small_object_memory_resource();

    ASSERT_NE(pool_mr, nullptr);
    ASSERT_EQ(pool_mr->upstream_resource(), mem::align::get_base_memory_resource());
}
