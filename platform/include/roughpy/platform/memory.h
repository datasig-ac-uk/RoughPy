//
// Created by sam on 24/10/24.
//

#ifndef ROUGHPY_PLATFORM_MEMORY_H
#define ROUGHPY_PLATFORM_MEMORY_H

#include "errors.h"

#include <atomic>
#include <limits>
#include <memory>
#include <memory_resource>
#include <utility>

#include <roughpy/core/alloc.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>

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
            dimn_t size,                                                       \
            dimn_t alignment                                                   \
    ) noexcept

#define RPY_FREE_FUNCTION(name)                                                \
    ROUGHPY_PLATFORM_EXPORT                                                    \
    void name(void* ptr, dimn_t size) noexcept

namespace dtl {

RPY_ALLOC_FUNCTION(aligned_alloc);
RPY_FREE_FUNCTION(aligned_free);

RPY_ALLOC_FUNCTION(small_object_alloc);
RPY_FREE_FUNCTION(small_object_free);

template <typename I>
constexpr enable_if_t<is_integral_v<I>, bool> is_alignment(I align)
{
    return align > 0 && (align & (align - 1)) == 0;
}

}// namespace dtl

RPY_NO_DISCARD ROUGHPY_PLATFORM_EXPORT std::pmr::memory_resource*
get_base_memory_resource() noexcept;
RPY_NO_DISCARD ROUGHPY_PLATFORM_EXPORT std::pmr::memory_resource*
get_small_object_memory_resource() noexcept;

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
            dtl::is_alignment(Alignment),
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
    RPY_NO_DISCARD pointer allocate(size_type bytes, const void* hint = nullptr)
    {
        if (bytes == 0) { return nullptr; }

        void* ptr = dtl::aligned_alloc(bytes * sizeof(Ty), alignment);
        if (!ptr) { throw std::bad_alloc(); }

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

class ROUGHPY_PLATFORM_EXPORT SmallObjectBase
{
public:
    void* operator new(dimn_t size);
    void operator delete(void* ptr, dimn_t size);
};

template <typename T>
class Rc
{
    T* p_data = nullptr;

public:
    using value_type = remove_reference_t<T>;
    using pointer = add_pointer_t<value_type>;
    using reference = add_lvalue_reference_t<T>;

    Rc(nullptr_t) noexcept : p_data(nullptr) {}

    Rc(pointer ptr)
    {
        RPY_DBG_ASSERT(ptr != nullptr);
        // TODO: Is this safe? It seems plausible that these two expressions
        // could be separated and thus the assignment to p_data does not
        // not happen, resulting in an indestructible object
        (p_data = ptr)->inc_ref();
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

    // ReSharper disable once CppNonExplicitConversionOperator
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

    constexpr pointer get() const noexcept { return p_data; }

    constexpr pointer release() noexcept
    {
        return std::exchange(p_data, nullptr);
    }

    constexpr void reset(pointer ptr = pointer()) noexcept
    {
        Rc(ptr).swap(*this);
    }

    void swap(Rc& other) noexcept { std::swap(p_data, other.p_data); }
};

template <typename T, typename... Args>
constexpr Rc<T> make_rc(Args&&... args)
{
    return Rc<T>(new T(std::forward<Args>(args)...));
}

class RcBase : public SmallObjectBase
{
    mutable std::atomic_intptr_t m_rc;

    template <typename Derived>
    friend Rc<Derived>;

protected:
    RcBase() = default;

    void inc_ref() const noexcept;
    bool dec_ref() const;
};

inline void RcBase::inc_ref() const
{
    auto old_rc = m_rc.fetch_add(1, std::memory_order_relaxed);
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
