//
// Created by sam on 24/10/24.
//


#include "memory.h"
#include "small_object_memory.h"


void* rpy::small::small_object_alloc(dimn_t size, dimn_t alignment) noexcept
{
    if (size == 0) { return nullptr; }
    return get_pool_memory()->allocate(size, alignment);
}
