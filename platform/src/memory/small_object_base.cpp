//
// Created by sam on 24/10/24.
//


#include "memory.h"

using namespace rpy;



void* SmallObjectBase::operator new(dimn_t size) {
    return small::small_object_alloc(size, small_alloc_chunk_size);
}

void SmallObjectBase::operator delete(void* p, dimn_t size) {
    small::small_object_free(p, size);
}

