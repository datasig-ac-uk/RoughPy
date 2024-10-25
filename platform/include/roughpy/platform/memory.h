//
// Created by sam on 24/10/24.
//

#ifndef ROUGHPY_PLATFORM_MEMORY_H
#define ROUGHPY_PLATFORM_MEMORY_H

#pragma once

#include <atomic>
#include <limits>
#include <memory>
#include <memory_resource>
#include <utility>

#include <roughpy/core/alloc.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>

#include "errors.h"
#include "roughpy/core/helpers.h"
#include "roughpy_platform_export.h"

namespace rpy {
/**
 * @brief The size of chunks allocated by the default small object allocator
 *
 *
 */
inline constexpr dimn_t small_alloc_chunk_size = 64;

/**
 * @brief Specifies the alignment requirement for data, in bytes.
 *
 * This constant defines the standard alignment, in bytes, for data
 * within the system. It is used to ensure that data structures and
 * allocations are aligned properly in memory to optimize performance
 * and avoid potential issues with misaligned access.
 */
inline constexpr dimn_t alloc_data_alignment = 64;

#define RPY_ALLOC_FUNCTION(name)                                               \
    RPY_NO_DISCARD ROUGHPY_PLATFORM_EXPORT void* name(                         \
            dimn_t alignment,                                                  \
            dimn_t size                                                        \
    ) noexcept
#define RPY_FREE_FUNCTION(name)                                                \
    ROUGHPY_PLATFORM_EXPORT                                                    \
    void name(void* ptr, dimn_t size) noexcept

namespace dtl {


/**
 * @brief Allocates memory with a specified alignment.
 *
 * This function allocates a block of memory of the given size, ensuring that
 * the block's starting address meets the specified alignment requirements.
 *
 * @param alignment The alignment requirement for the memory block. Must be a
 * power of two.
 * @param size The size of the memory block to allocate.
 * @return A pointer to the allocated memory block, or nullptr if the allocation
 * fails.
 */
RPY_ALLOC_FUNCTION(aligned_alloc);


/**
 * @brief Frees memory allocated with alignment.
 *
 * This function releases a block of memory that was previously allocated with
 * alignment requirements. The behavior is implementation-defined depending on
 * the operating system and compiler being used.
 *
 * @param ptr A pointer to the memory block to free.
 * @param size The size of the memory block. This parameter is ignored by the
 * function.
 * @return void
 */
RPY_FREE_FUNCTION(aligned_free);


/**
 * @brief Allocates a small object with the specified size and alignment.
 *
 * This function allocates a small memory object of the given size, ensuring
 * that the object's starting address meets the specified alignment
 * requirements. If the size is zero, the function returns nullptr.
 *
 * @param alignment The alignment requirement for the small object. Must be a
 * power of two.
 * @param size The size of the small object to allocate.
 * @return A pointer to the allocated small object, or nullptr if the allocation
 * fails or if size is zero.
 */
RPY_ALLOC_FUNCTION(small_object_alloc);


/**
 * @brief Frees a small object of a specified size.
 *
 * This function releases a block of memory that was previously allocated for a
 * small object. The behavior is implementation-defined depending on the
 * operating system and compiler being used.
 *
 * @param ptr A pointer to the small object memory block to free.
 * @param size The size of the small object memory block. This parameter is used
 * to facilitate the memory free operation.
 */
RPY_FREE_FUNCTION(small_object_free);



}// namespace dtl


namespace align {

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
constexpr bool is_pointer_aligned(const volatile void* ptr, std::size_t alignment)
{
    RPY_DBG_ASSERT(is_alignment(alignment));
    return bit_cast<std::uintptr_t>(ptr) % alignment == 0;
}


}


/**
 * @brief Retrieves the base memory resource for the current allocations
 *
 * This function provides access to the base memory resource, which is used as
 * the underlying allocator for memory management operations. It is typically
 * employed in scenarios where custom memory resources or allocators are in
 * play, allowing adjustments or inquiries regarding the foundational memory
 * handling mechanism.
 *
 * @return The base memory resource currently in use
 */
RPY_NO_DISCARD ROUGHPY_PLATFORM_EXPORT std::pmr::memory_resource*
get_base_memory_resource() noexcept;

/**
 * @brief Retrieves the memory resource used for small object allocations.
 *
 * This function returns the memory resource that is utilized for the allocation
 * of small objects, enhancing memory management efficiency for frequently
 * created small objects.
 *
 * @return A pointer to the memory resource used for small object allocations.
 */
RPY_NO_DISCARD ROUGHPY_PLATFORM_EXPORT std::pmr::memory_resource*
get_small_object_memory_resource() noexcept;

/**
 * @brief Retrieves the default allocator instance.
 *
 * This function provides access to the default allocator,
 * which is responsible for managing memory allocations
 * and deallocations in a standardized manner.
 *
 * @return A pointer to the default allocator instance.
 */
template <typename Ty>
std::pmr::polymorphic_allocator<Ty> get_default_allocator() noexcept
{
    return std::pmr::polymorphic_allocator<Ty>(get_base_memory_resource());
}


/**
 * @brief Retrieves the small object allocator.
 *
 * This method provides access to a specialized allocator designed
 * for efficiently managing small objects.
 *
 * @return The small object allocator.
 */
template <typename Ty>
std::pmr::polymorphic_allocator<Ty> get_small_object_allocator() noexcept
{
    return std::pmr::polymorphic_allocator<Ty>(get_small_object_memory_resource(
    ));
}

/**
 * @class AlignedAllocator
 * @brief A custom allocator that provides memory allocation with specified
 * alignment.
 *
 * AlignedAllocator is designed for scenarios where memory alignment is crucial,
 * such as SIMD operations or hardware-specific data structures. It ensures that
 * all allocated memory blocks meet the specified alignment requirements.
 *
 * @tparam Ty The type of objects to be allocated.
 * @tparam Alignment The alignment in bytes for the memory allocation.
 */
template <typename Ty, size_t Alignment = alloc_data_alignment>
class AlignedAllocator
{
    static_assert(
            align::is_alignment(Alignment),
            "Valid alignments are powers of 2 and greater than 0"
    );

public:
    // Type definitions required for the allocator interface
    using value_type = Ty;
    using pointer = Ty*;
    using const_pointer = const Ty*;
    using reference = Ty&;
    using const_reference = const Ty&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    static constexpr size_t alignment
            = std::max(alignof(Ty), alloc_data_alignment);

    // Rebind allocator to another type
    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    // Default constructor
    AlignedAllocator() noexcept = default;

    // Copy constructor
    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept
    {}

    // Address of
    RPY_NO_DISCARD pointer address(reference x) const noexcept
    {
        return std::addressof(x);
    }

    RPY_NO_DISCARD const_pointer address(const_reference x) const noexcept
    {
        return std::addressof(x);
    }

    // Allocate memory
    RPY_NO_DISCARD pointer allocate(size_type size, const void* hint = nullptr)
    {
        if (RPY_UNLIKELY(size == 0)) { return nullptr; }

        void* ptr = dtl::aligned_alloc(alignment, size * sizeof(Ty));
        if (RPY_UNLIKELY(!ptr)) { throw std::bad_alloc(); }

        return static_cast<pointer>(ptr);
    }

    // Deallocate memory
    void deallocate(pointer p, size_type n) noexcept
    {
        dtl::aligned_free(p, n * sizeof(Ty));
    }

    // Maximum size
    RPY_NO_DISCARD size_type max_size() const noexcept
    {
        return (std::numeric_limits<size_type>::max() - alignment) / sizeof(Ty);
    }

    // Construct an object in-place
    template <typename U, typename... Args>
    void construct(U* p, Args&&... args)
    {
        construct_inplace(p, std::forward<Args>(args)...);
    }

    // Destroy an object in-place
    template <typename U>
    void destroy(U* p)
    {
        p->~U();
    }
};

template <typename Ty, size_t TyAlign, typename Uy, size_t UyAlign>
RPY_NO_DISCARD constexpr bool operator==(
        const AlignedAllocator<Ty, TyAlign>& RPY_UNUSED_VAR left,
        AlignedAllocator<Uy, UyAlign>& RPY_UNUSED_VAR right
) noexcept
{
    return TyAlign == UyAlign;
}

template <typename Ty, size_t TyAlign, typename Uy, size_t UyAlign>
RPY_NO_DISCARD constexpr bool operator!=(
        const AlignedAllocator<Ty, TyAlign>& RPY_UNUSED_VAR left,
        AlignedAllocator<Uy, UyAlign>& RPY_UNUSED_VAR right
) noexcept
{
    return TyAlign != UyAlign;
}

/**
 * @brief Base class for small object allocation.
 *
 * Provides a base class for objects that utilize a custom
 * allocator optimized for small objects.
 *
 * This class typically serves as a base for other classes
 * to inherit from, enabling efficient memory management
 * for small-sized objects.
 */
class ROUGHPY_PLATFORM_EXPORT SmallObjectBase
{
public:
    void* operator new(dimn_t size);
    void operator delete(void* ptr, dimn_t size);
};


/*
 * Reference counting and resource management
 *
 * We will make extensive use of reference counted smart pointers throughout
 * RoughPy. The std::shared_ptr type is ok, but it has a very important downside
 * is the size of two pointers, not just one. This is a problem because quite a
 * lot of our uses will require a ref-counted pointer alongside some other
 * pointer, which would mean a total size of three pointers. This has some
 * fairly major consequences, especially when these objects appear in arrays.
 *
 * The alternative is a boost intrusive_ptr style reference counter, which
 * relies on the object itself to provide the reference counting mechanism. This
 * is great because it allows the smart pointer to be the size of one pointer,
 * but it also has a problem. It uses a complicated mechanism of ADL external
 * functions to increase and decrease the reference count. This is frustrating,
 * because it means these functions have to be implemented whenever the class
 * should support reference counting. (This is made easier with a base class,
 * but this lacks flexibility in special cases.)
 *
 * We turn to Google's TSL for some inspiration. They use member functions to
 * on the class to manage the reference count. This is more flexible because it
 * allows us to make use of virtual functions in some special cases to customise
 * the reference counting whilst still making use of the convenience of the base
 * class and still having the one-pointer-sized smart pointer that we really
 * need.
 *
 */


// template <typename T>
// concept ReferenceCountable = requires(const T& type)
// {
//     { type.inc_ref() } -> std::convertible_to<void>;
//     { type.dec_ref() } -> std::convertible_to<bool>;
//     { type.ref_count() } -> std::convertible_to<dimn_t>;
// };




/**
 * @brief A reference-counted class for resource management
 *
 * This class implements a simple reference counting mechanism which helps
 * in managing the lifetime of objects. It ensures that the resource is
 * properly released when there are no more references to it.
 *
 * Rc class maintains a count of the number of references to a resource and
 * automatically frees the resource when the count reaches zero.
 *
 * @tparam T Object type pointed to by pointer, should satisfy the
 * ReferenceCountable concept
 */
template <typename T>
class Rc
{
public:
    using value_type = remove_reference_t<T>;
    using pointer = add_pointer_t<value_type>;
    using reference = add_lvalue_reference_t<value_type>;

private:
    T* p_data = nullptr;

public:

    Rc(std::nullptr_t) noexcept : p_data(nullptr) {}

    Rc(pointer ptr)
    {
        if (ptr != nullptr) {
            // It seems plausible that these two expressions
            // could be separated and thus the assignment to p_data does
            // not happen, resulting in an indestructible object
            (p_data = ptr)->inc_ref();
        }
    }

    Rc(const Rc& other) : Rc(other.get()) {}

    Rc(Rc&& other) noexcept : p_data(other.release()) {}

    ~Rc()
    {
        if (p_data) { p_data->dec_ref(); }
    }

    Rc& operator=(const Rc& other)
    {
        if (this != &other) { Rc(other).swap(*this); }
        return *this;
    }

    Rc& operator=(Rc&& other) noexcept
    {
        if (this != &other) { this->reset(other.release()); }
        return *this;
    }

    /**
     * @brief Overloads an operator for the class
     *
     * This function provides a custom implementation of an operator for the
     * class. The specific operator and its behavior should be detailed in the
     * function.
     */
    constexpr operator bool() const noexcept { return p_data != nullptr; }

    constexpr pointer operator->() const noexcept
    {
        RPY_DBG_ASSERT(*this);
        return p_data;
    }

    constexpr reference operator*() const noexcept
    {
        RPY_DBG_ASSERT(*this);
        return *p_data;
    }

    /**
     * @brief Retrieves the value associated with the specified key from a
     * container
     *
     * This method searches for the given key in a container and returns the
     * corresponding value if found.
     *
     * @return The value associated with the specified key, or an indication
     * that the key was not found
     */
    constexpr pointer get() const noexcept { return p_data; }

    /**
     * @brief Releases the contained pointer from management
     *
     * This replaces the managed pointed with nullptr and returns the managed
     * pointer without decrementing the reference count. The primary use is to
     * implement move semantics.
     *
     * @return the previously contained pointer
     */
    constexpr pointer release() noexcept
    {
        return std::exchange(p_data, nullptr);
    }

    /**
     * @brief Resets the Rc object to manage a new pointer, releasing any
     * currently managed object.
     *
     * This method assigns the provided pointer to the current Rc object,
     * releasing the ownership of any previously managed resource. If no pointer
     * is provided, it defaults to managing a nullptr. The Rc object will then
     * manage the newly assigned pointer.
     *
     * @param ptr The new pointer to be managed by this Rc object. Defaults to
     * nullptr if not provided.
     */
    constexpr void reset(pointer ptr = pointer()) noexcept
    {
        Rc(ptr).swap(*this);
    }

    /**
     * @brief Swaps the managed pointers of with another Rc
     *
     * @param other the Rc to swap managed pointer with
     */
    void swap(Rc& other) noexcept { std::swap(p_data, other.p_data); }
};


/**
 * @brief Creates a reference-counted smart pointer
 *
 * @tparam T The type of the object to manage
 * @param args The arguments to be forwarded to the constructor of T
 * @return A std::shared_ptr<T> managing the newly created object
 */
template <typename T, typename... Args>
constexpr Rc<T> make_rc(Args&&... args)
{
    return Rc<T>(new T(std::forward<Args>(args)...));
}


/**
 * @brief A base class providing reference counting functionality.
 *
 * This class offers fundamental methods for managing the reference count,
 * suitable for derived classes that need reference-counted behavior.
 */
class RcBase : public SmallObjectBase
{
    mutable std::atomic_intptr_t m_rc;

    template <typename Derived>
    friend class Rc;

protected:
    RcBase() = default;

    /**
     * @brief Increments the reference count of the RcBase object
     *
     * This method is used to increase the reference count of the RcBase object
     * ensuring that the object is properly tracked for reference management.
     */
    void inc_ref() const noexcept;

    /**
     * @brief Decreases the reference count of the object
     *
     * This method decrements the reference count of the object and deletes
     * it if the reference count reaches zero.
     *
     * @return true if the object was deleted, otherwise false
     */
    bool dec_ref() const;

    /**
     * @brief The reference count for a shared object
     *
     * This value represents the number of references currently held to a shared
     * object. When the count reaches zero, the shared object can be safely
     * deleted.
     *
     * @return The current reference count
     */
    RPY_NO_DISCARD dimn_t ref_count() const noexcept
    {
        return static_cast<dimn_t>(m_rc.load(std::memory_order_acquire));
    }
};

inline void RcBase::inc_ref() const noexcept
{
    auto old_rc = m_rc.fetch_add(1, std::memory_order_relaxed);
    ignore_unused(old_rc);
    RPY_DBG_ASSERT(old_rc > 0);
}

inline bool RcBase::dec_ref() const
{
    RPY_DBG_ASSERT(m_rc.load() > 0);
    if (m_rc.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        delete this;
        return true;
    }
    return false;
}

}// namespace rpy

#endif// ROUGHPY_PLATFORM_MEMORY_H
