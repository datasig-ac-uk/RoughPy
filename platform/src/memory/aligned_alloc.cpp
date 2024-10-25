//
// Created by sam on 24/10/24.
//


#include "memory.h"

#include <cstdlib>

#include <roughpy/core/macros.h>


void* rpy::align::aligned_alloc(dimn_t alignment, dimn_t size) noexcept
{
    if (RPY_UNLIKELY(size == 0)) { return nullptr; }
    if (RPY_UNLIKELY(alignment == 0)) { alignment = alignof(std::max_align_t); }
    RPY_DBG_ASSERT(align::is_alignment(alignment));

#ifdef RPY_MSVC
    return _aligned_malloc(size, size);
#elif defined(RPY_GCC) || defined(RPY_CLANG)
    void *ptr;
    if (int err = posix_memalign(&ptr, std::max(alignment, sizeof(void*)), size) != 0) {
        ignore_unused(err);
        ptr = nullptr;
    }
    return ptr;
#endif
}

void rpy::align::aligned_free(void* ptr, dimn_t size) noexcept
{
    ignore_unused(size);
#ifdef RPY_MSVC
    _aligned_free(ptr);
#elif defined(RPY_GCC) || defined(RPY_CLANG)
    free(ptr);
#endif
}



