//
// Created by sam on 25/10/24.
//

#include <gtest/gtest.h>

#include <roughpy/platform/memory.h>


using namespace rpy;

TEST(SmalLObjectAllocatorTest, TestUpstreamAllocatorSetCorrectly)
{
    auto* small_object_mr = get_small_object_memory_resource();

    auto* pool_mr = dynamic_cast<std::pmr::synchronized_pool_resource*>(small_object_mr);
    ASSERT_NE(pool_mr, nullptr);

    ASSERT_EQ(pool_mr->upstream_resource(), get_base_memory_resource());
}
