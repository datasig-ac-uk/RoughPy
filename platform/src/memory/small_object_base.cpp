//
// Created by sam on 24/10/24.
//


#include "memory.h"
#include "small_object_memory.h"

using namespace rpy;



void* mem::SmallObjectBase::operator new(dimn_t size) {
    if (RPY_UNLIKELY(size == 0)) { return nullptr; }
    auto* pool = get_pool_memory();
    RPY_DBG_ASSERT(pool != nullptr);

    void* ptr = pool->allocate(size, alignof(std::max_align_t));
    if (RPY_UNLIKELY(ptr == nullptr)) {
        throw std::bad_alloc();
    }
    return ptr;
}

void mem::SmallObjectBase::operator delete(void* p, dimn_t size) {
    auto* pool = get_pool_memory();
    RPY_DBG_ASSERT(pool != nullptr);
    pool->deallocate(p, size);
}

