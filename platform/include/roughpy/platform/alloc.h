//
// Created by sammorley on 19/11/24.
//

#ifndef ALLOC_H
#define ALLOC_H

#include "roughpy/core/macros.h"
#include "roughpy/core/types.h"

#include "roughpy/platform/roughpy_platform_export.h"

namespace rpy::mem {


// 4K is a typical page size most operating systems
inline constexpr size_t small_chunk_size = 4096;
inline constexpr size_t small_block_size = 64;
inline constexpr size_t small_blocks_per_chunk = small_chunk_size / small_block_size;

RPY_NO_DISCARD ROUGHPY_PLATFORM_EXPORT void*
aligned_alloc(size_t alignment, size_t size) noexcept;

ROUGHPY_PLATFORM_EXPORT void aligned_free(void* ptr, size_t size=0) noexcept;

RPY_NO_DISCARD ROUGHPY_PLATFORM_EXPORT
void* small_object_alloc(size_t size);

ROUGHPY_PLATFORM_EXPORT
void small_object_free(void* ptr, size_t size);


class ROUGHPY_PLATFORM_EXPORT SmallObjectBase
{
public:
    void* operator new(size_t size);
    void operator delete(void* object, size_t size);
};

}// namespace rpy::mem

#endif// ALLOC_H
