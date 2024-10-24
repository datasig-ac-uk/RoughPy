//
// Created by sam on 24/10/24.
//


#include "memory.h"

#ifdef RPY_MSVC
#include <cstdlib>
#endif

#include <roughpy/core/macros.h>


void* rpy::dtl::aligned_alloc(dimn_t size, dimn_t alignment) noexcept
{
    if (size == 0) { return nullptr; }

#ifdef RPY_MSVC
    return _aligned_malloc(alignment, size);
#else if defined(RPY_GCC) || defined(RPY_CLANG)
    void *ptr;
    if (posix_memalign(&ptr, size, alignment) != 0) {
        ptr = nullptr;
    }
    return ptr;
#endif
}

void rpy::dtl::aligned_free(void* ptr, dimn_t size) noexcept
{
    ignore_unused(size);
#ifdef RPY_MSVC
    _aligned_free(ptr);
#else if defined(RPY_GCC) || defined(RPY_CLANG)
    free(ptr);
#endif
}



