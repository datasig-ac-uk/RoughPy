//
// Created by sam on 24/10/24.
//


#include "memory.h"
#include "small_object_memory.h"


void* rpy::small::small_object_alloc(dimn_t alignment, dimn_t size) noexcept
{
    if (RPY_UNLIKELY(size == 0)) { return nullptr; }
    return get_pool_memory()->allocate(size, alignment);
}

void rpy::small::small_object_free(void* ptr, dimn_t size) noexcept
{
    get_pool_memory()->deallocate(ptr, size);
}