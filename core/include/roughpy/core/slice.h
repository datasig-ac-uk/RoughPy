#ifndef ROUGHPY_CORE_SLICE_H_
#define ROUGHPY_CORE_SLICE_H_

#include "implementation_types.h"
#include "traits.h"

#include <iterator>
#include <vector>

namespace rpy {

/**
 * @brief Common access for contiguous array-like data
 *
 * A slice is a view into a contiguous block of data, such as a
 * C array or a C++ vector. This provides a common surface for
 * accepting all such arguments without having to take raw pointer/
 * size pairs as arguments. The implicit conversion from common
 * data types means that one will rarely need to think about the
 * actual container.
 *
 * @tparam T Type of data
 */
template <typename T>
class Slice {
    T *p_data = nullptr;
    std::size_t m_size = 0;

public:
    constexpr Slice() = default;

    constexpr Slice(T &num) : p_data(&num), m_size(1) {}

    constexpr Slice(nullptr_t) : p_data(nullptr), m_size(0) {}

    template <typename Container, typename = traits::enable_if_t<traits::is_same<typename Container::value_type, T>::value>>
    constexpr Slice(Container &container)
        : p_data(container.data()), m_size(container.size()) {}

    template <std::size_t N>
    constexpr Slice(T (&array)[N])
        : p_data(array), m_size(N) {}

    constexpr Slice(T *ptr, std::size_t N)
        : p_data(ptr), m_size(N) {}

    template <typename I>
    constexpr traits::enable_if_t<
        traits::is_integral<I>::value,
        const T &>
    operator[](I i) noexcept {
        assert(0 <= i && i < m_size);
        return p_data[i];
    }

    template <typename I>
    constexpr traits::enable_if_t<
        traits::is_integral<I>::value,
        T &>
    operator[](I i) const noexcept {
        assert(0 <= i && i < m_size);
        return p_data[i];
    }

    constexpr bool empty() const noexcept { return p_data == nullptr || m_size == 0; }

    constexpr std::size_t size() const noexcept {
        return m_size;
    }

    constexpr T *begin() noexcept { return p_data; }
    constexpr T *end() noexcept { return p_data + m_size; }
    constexpr const T *begin() const { return p_data; }
    constexpr const T *end() const { return p_data + m_size; }

    constexpr std::reverse_iterator<T *> rbegin() noexcept { return std::reverse_iterator<T *>(p_data + m_size); }
    constexpr std::reverse_iterator<T *> rend() noexcept { return std::reverse_iterator<T *>(p_data); }
    constexpr std::reverse_iterator<const T *> rbegin() const noexcept { return std::reverse_iterator<const T *>(p_data + m_size); }
    constexpr std::reverse_iterator<const T *> rend() const noexcept { return std::reverse_iterator<const T *>(p_data); }

    operator std::vector<T>() const {
        std::vector<T> result;
        result.reserve(m_size);
        for (dimn_t i = 0; i < m_size; ++i) {
            result.push_back(p_data[i]);
        }
        return result;
    }
};
}// namespace rpy

#endif// ROUGHPY_CORE_SLICE_H_
