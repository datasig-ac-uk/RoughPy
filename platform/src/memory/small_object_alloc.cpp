//
// Created by sam on 24/10/24.
//


#include "memory.h"
#include "small_object_memory.h"

using namespace rpy;

void* mem::small::small_object_alloc(dimn_t alignment, dimn_t size) noexcept
{
    if (RPY_UNLIKELY(size == 0)) { return nullptr; }
    auto* pool = get_pool_memory();
    RPY_DBG_ASSERT(pool != nullptr);
    return pool->allocate(size, alignment);
}

void mem::small::small_object_free(void* ptr, dimn_t size) noexcept
{
    auto* pool = get_pool_memory();
    RPY_DBG_ASSERT(pool != nullptr);
    pool->deallocate(ptr, size);
}