#ifndef ROUGHPY_COMPUTE_COMMON_CACHE_ARRAY_H
#define ROUGHPY_COMPUTE_COMMON_CACHE_ARRAY_H

#include <cstdint>
#include <cassert>
#include <memory>
#include <algorithm>


namespace rpy::compute {

/**
 * @class CacheArray
 * @brief A lightweight array container optimized for small sizes to fit into low-level CPU caches.
 *
 * The CacheArray class is designed to minimize dynamic memory allocations by using
 * a pre-allocated inline buffer for small arrays. For larger arrays exceeding this
 * inline buffer size, dynamic memory allocation is used.
 *
 * @tparam T The type of elements stored in the array.
 * @tparam InlineSize The size of the pre-allocated inline buffer. Arrays with this size
 *         or smaller will avoid heap allocation.
 * @tparam Allocator_ The allocator type to use for dynamic memory allocation,
 *         defaults to `std::allocator<T>`.
 */
template <typename T, std::size_t InlineSize=sizeof(std::size_t)/sizeof(T), typename Allocator_=std::allocator<T>>
class CacheArray : Allocator_ {
    using Traits = std::allocator_traits<Allocator_>;

    union {
        T inline_buffer_[InlineSize];
        std::size_t alloc_;
    };
    T* ptr_;
    std::size_t size_;

    constexpr bool is_inline() const noexcept {
        return size_ <= InlineSize;
    }

public:

    explicit CacheArray(std::size_t size) : size_(size){
        if (size > InlineSize) {
            ptr_ = Traits::allocate(*this, size);
            alloc_ = size;
        } else {
            ptr_ = inline_buffer_;
        }
    }

    ~CacheArray() {
        if (!is_inline()) {
            Traits::deallocate(*this, ptr_, size_);
        }
    }

    [[nodiscard]]
    constexpr T const& operator[](std::size_t index) const noexcept {
        assert(ptr_ != nullptr && index < size_);
        return ptr_[index];
    }

    [[nodiscard]]
    constexpr T& operator[](std::size_t index) noexcept {
        assert(ptr_ != nullptr && index < size_);
        return ptr_[index];
    }

    [[nodiscard]]
    constexpr std::size_t size() const noexcept { return size_; }

    [[nodiscard]]
    constexpr std::size_t capacity() const noexcept {
        return is_inline() ? InlineSize : alloc_;
    }

    [[nodiscard]]
    constexpr T* data() noexcept { return ptr_; }

    [[nodiscard]]
    constexpr T const* data() const noexcept { return ptr_; }

    /**
     * @brief Resize the array.
     *
     * This changes the internal size of the array, causing a new allocation if
     * the old allocation is not large enough to hold the new size. This method
     * invalidates the entries of the array, and the array should be considered
     * empty and should be reinitialized after this operation.
     *
     * @param new_size new size that the array should occupy.
     */
    void resize(std::size_t new_size) {
        if (is_inline()) {
            if (new_size > InlineSize) {
                ptr_ = Traits::allocate(*this, new_size);
                alloc_ = new_size;
                size_ = new_size;
            }
        } else {
            if (new_size > alloc_) {
                Traits::deallocate(*this, ptr_, alloc_);
                ptr_ = Traits::allocate(*this, new_size);
                size_ = new_size;
                alloc_ = new_size;
            } else {
                size_ = new_size;
            }
        }
    }

};


} // namespace rpy::compute



#endif //ROUGHPY_COMPUTE_COMMON_CACHE_ARRAY_H
