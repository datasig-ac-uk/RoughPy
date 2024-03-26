//
// Created by sam on 3/13/24.
//

#ifndef ROUGHPY_ALLOC_H
#define ROUGHPY_ALLOC_H

#include "roughpy_platform_export.h"

#include <roughpy/core/alloc.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include <memory>

namespace rpy {
namespace platform {

/// the size of chunks allocated by the small object allocator
static constexpr dimn_t small_alloc_chunk_size = 64;

/**
 * @brief Fast pool allocator for small objects
 * @param size number of bytes to allocate
 * @return pointer to newly allocated block
 *
 *
 * This allocator will try to allocate from a pool of chunks. failing that, it
 * will use malloc to allocate from the heap
 */
RPY_NO_DISCARD void* alloc_small(dimn_t size);

/**
 * @brief Free a small object allocated with alloc_small
 */
void free_small(void* ptr, dimn_t size);

class ROUGHPY_PLATFORM_EXPORT SmallObjectBase
{
public:
    void* operator new(dimn_t size);
    void operator delete(void* ptr, dimn_t size);
};

}// namespace platform
}// namespace rpy

#endif// ROUGHPY_ALLOC_H
