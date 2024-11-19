#ifndef ROUGHPY_CORE_ALLOC_H_
#define ROUGHPY_CORE_ALLOC_H_

#include "traits.h"

namespace rpy {

template <typename T, typename... Args>
constexpr void construct_inplace(
        T* dst,
        Args&&... args
) noexcept(is_nothrow_constructible_v<T, Args...>)
{
    ::new (static_cast<void*>(dst)) T(std::forward<Args>(args)...);
}

template <typename T, typename... Args>
constexpr void construct_inplace(
        void* dst,
        Args&&... args
) noexcept(is_nothrow_constructible_v<T, Args...>)
{
    ::new (dst) T(std::forward<Args>(args)...);
}

template <typename T, typename... Args>
constexpr void construct_inplace(
        T& dst,
        Args&&... args
) noexcept(is_nothrow_constructible_v<T, Args...>)
{
    ::new (static_cast<void*>(std::addressof(dst)))
            T(std::forward<Args>(args)...);
}

}// namespace rpy

#endif// ROUGHPY_CORE_ALLOC_H_
