//
// Created by sam on 3/13/24.
//

#include "alloc.h"

#include <memory>
#include <memory_resource>

using namespace rpy;
using namespace rpy::platform;

static std::pmr::synchronized_pool_resource
        s_resource({4096 / small_alloc_chunk_size, small_alloc_chunk_size});

void* rpy::platform::alloc_small(rpy::dimn_t size)
{
    return s_resource.allocate(size);
}

void rpy::platform::free_small(void* ptr, dimn_t size)
{
    s_resource.deallocate(ptr, size);
}

void* SmallObjectBase::operator new(dimn_t size) {
    return alloc_small(size);
}
void SmallObjectBase::operator delete(void* ptr, dimn_t size) {
    free_small(ptr, size);
}
