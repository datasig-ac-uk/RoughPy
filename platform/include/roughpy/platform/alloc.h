//
// Created by sammorley on 19/11/24.
//

#ifndef ALLOC_H
#define ALLOC_H

#include "roughpy/core/debug_assertion.h"
#include "roughpy/core/macros.h"
#include "roughpy/core/traits.h"
#include "roughpy/core/types.h"

#include "roughpy/platform/roughpy_platform_export.h"

namespace rpy::mem {

// 4K is a typical page size most operating systems
inline constexpr size_t small_chunk_size = 4096;
inline constexpr size_t small_block_size = 64;
inline constexpr size_t small_blocks_per_chunk
        = small_chunk_size / small_block_size;

/**
 * @brief Allocates a block of aligned memory.
 *
 * This function allocates a block of memory of the specified size with the
 * specified alignment. The alignment must be a power of two, and the size must
 * be a multiple of the alignment.
 *
 * @param alignment The required alignment of the allocated memory block.
 * @param size The size of the memory block to allocate, in bytes.
 * @return A pointer to the allocated memory block, or nullptr if the allocation
 * fails.
 */
RPY_NO_DISCARD ROUGHPY_PLATFORM_EXPORT void*
aligned_alloc(size_t alignment, size_t size) noexcept;

/**
 * @brief Frees a block of memory that was aligned and allocated.
 *
 * This function releases a block of memory that was previously allocated
 * with an alignment requirement. It ensures that the memory is properly
 * deallocated and the alignment constraints are respected.
 *
 * @param ptr A pointer to the memory block to be freed.
 */
ROUGHPY_PLATFORM_EXPORT void aligned_free(void* ptr, size_t size = 0) noexcept;

/**
 * @brief Allocates memory for a small object.
 *
 * This function allocates a block of memory of the specified size
 * using a synchronized pool resource to optimize allocation for small objects.
 *
 * @param size The size of the memory block to allocate, in bytes.
 * @return A pointer to the allocated memory block.
 */
RPY_NO_DISCARD ROUGHPY_PLATFORM_EXPORT void* small_object_alloc(size_t size);

/**
 * @brief Frees memory allocated for a small object.
 *
 * This function is responsible for deallocating memory that was previously
 * allocated for a small object. Ensures that resources used by the object
 * are properly released.
 *
 * @param ptr The pointer to the small object to be freed.
 */
ROUGHPY_PLATFORM_EXPORT
void small_object_free(void* ptr, size_t size);

/**
 * @brief Base class for small object optimization.
 *
 * This class provides foundational functionalities for small objects,
 * including memory management optimizations to reduce heap allocations.
 * Objects inheriting from this class may benefit from improved performance
 * for scenarios involving frequent creation and destruction of small objects.
 */
class ROUGHPY_PLATFORM_EXPORT SmallObjectBase
{
public:
    void* operator new(size_t size);
    void operator delete(void* object, size_t size);
};

/**
 * @brief Checks if a given value is a valid alignment.
 *
 * This function determines whether the specified alignment value is valid.
 * An alignment value is considered valid if it is a positive integer
 * and a power of two.
 *
 * @param align The alignment value to check.
 * @tparam I Integral type of the alignment value.
 * @return True if the alignment value is valid, false otherwise.
 */
template <typename I>
constexpr enable_if_t<is_integral_v<I>, bool> is_alignment(I align)
{
    return align > 0 && (align & (align - 1)) == 0;
}

/**
 * @brief Checks if a given pointer is aligned to a specified boundary.
 *
 * This function determines whether the specified pointer address is aligned
 * to the given boundary. The boundary must be a power of two.
 *
 * @param ptr The pointer to check.
 * @param alignment The alignment boundary to check against. Must be a power of
 * two.
 * @return True if the pointer is aligned to the specified boundary, false
 * otherwise.
 */
inline bool is_pointer_aligned(const volatile void* ptr, std::size_t alignment)
{
    // TODO: This function should be constexpr, we need to make use of constexpr
    //       bit cast or something similar.
    RPY_DBG_ASSERT(is_alignment(alignment));
    return reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0;
}



struct AlignedAllocHelper
{
    static void* allocate(size_t alignment, size_t size) noexcept
    {
        return aligned_alloc(alignment, size);
    }

    static void free(void* ptr, size_t size) noexcept
    {
        aligned_free(ptr, size);
    }
};




template <typename AllocHelper=AlignedAllocHelper>
class ScopedSafePtr
{
    void* p_data;
    size_t m_size;

public:
    ScopedSafePtr(size_t size, size_t alignment)
        : p_data(AllocHelper::allocate(alignment, size)),
          m_size(size)
    {}

    ~ScopedSafePtr()
    {
        if (p_data != nullptr) { AllocHelper::free(p_data, m_size); }
    }

    void* data() noexcept
    {
        return p_data;
    }

    void reset() noexcept
    {
        p_data = nullptr;
        m_size = 0;
    }
};

}// namespace rpy::mem

#endif// ALLOC_H
