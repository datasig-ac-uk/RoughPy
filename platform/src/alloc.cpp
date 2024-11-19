//
// Created by sammorley on 19/11/24.
//

#include "roughpy/platform/alloc.h"

#include <boost/align/aligned_alloc.hpp>

using namespace rpy::mem;
void* rpy::mem::aligned_alloc(size_t alignment, size_t size)
{
    return boost::alignment::aligned_alloc(alignment, size);
}

void rpy::mem::aligned_free(void* ptr)
{
    boost::alignment::aligned_free(ptr);
}
