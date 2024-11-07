//
// Created by user on 07/02/23.
//

#ifndef LIBALGEBRA_LITE_INCLUDE_LIBALGEBRA_LITE_DETAIL_NOTNULL_H
#define LIBALGEBRA_LITE_INCLUDE_LIBALGEBRA_LITE_DETAIL_NOTNULL_H


#include <cassert>
#include <memory>

namespace lal {
namespace dtl {

template <typename T>
class not_null
{
    T* p_data;

public:

    not_null() = delete;

    constexpr explicit not_null(T* ptr) : p_data(ptr)
    {
        assert(ptr != nullptr);
    }

    constexpr explicit not_null(const std::unique_ptr<T>& ptr) : p_data(ptr.get()) {
        assert(static_cast<bool>(ptr));
    }

    not_null(std::nullptr_t) = delete;
    constexpr not_null(const not_null&) = default;
    constexpr not_null(not_null&&) noexcept = default;

    constexpr operator T*() const noexcept { return p_data; }

    constexpr not_null& operator=(const not_null&) noexcept = default;
    constexpr not_null& operator=(not_null&&) noexcept = default;

    constexpr not_null& operator=(T* other) noexcept
    {
        if (other != p_data) {
            p_data = other;
        }
        return *this;
    }

    constexpr not_null& operator=(std::nullptr_t) = delete;


    constexpr T* operator->() const noexcept { return p_data; }
    constexpr T& operator*() const noexcept { return *p_data; }

};


} // namespace dtl
} // namespace lal



#endif //LIBALGEBRA_LITE_INCLUDE_LIBALGEBRA_LITE_DETAIL_NOTNULL_H
