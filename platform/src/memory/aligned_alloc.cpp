//
// Created by sam on 24/10/24.
//


#include "memory.h"

#ifdef RPY_MSVC
#include <cstdlib>
#endif

#include <roughpy/core/macros.h>


void* rpy::dtl::aligned_alloc(dimn_t alignment, dimn_t size) noexcept
{
    if (RPY_UNLIKELY(size == 0)) { return nullptr; }
    if (RPY_UNLIKELY(alignment == 0)) { alignment = alignof(std::max_align_t); }
    RPY_DBG_ASSERT(align::is_alignment(alignment));

#ifdef RPY_MSVC
    return _aligned_malloc(size, size);
#elif defined(RPY_GCC) || defined(RPY_CLANG)
    void *ptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
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
#elif defined(RPY_GCC) || defined(RPY_CLANG)
    free(ptr);
#endif
}



