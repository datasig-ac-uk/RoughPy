// Copyright (c) 2023 RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef ROUGHPY_CORE_SLICE_H_
#define ROUGHPY_CORE_SLICE_H_

#include "traits.h"
#include "types.h"

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
class Slice
{
    T* p_data = nullptr;
    std::size_t m_size = 0;

public:
    constexpr Slice() = default;

    constexpr Slice(T& num) : p_data(&num), m_size(1) {}

    constexpr Slice(std::nullptr_t) : p_data(nullptr), m_size(0) {}

    template <
            typename Container,
            typename
            = enable_if_t<is_same<typename Container::value_type, T>::value>>
    constexpr Slice(Container& container)
        : p_data(container.data()), m_size(container.size())
    {}

    template <std::size_t N>
    constexpr Slice(T (&array)[N]) : p_data(array), m_size(N)
    {}

    constexpr Slice(T* ptr, std::size_t N) : p_data(ptr), m_size(N) {}

    template <typename I>
    constexpr enable_if_t<is_integral<I>::value, const T&> operator[](I i
    ) noexcept
    {
        RPY_DBG_ASSERT(0 <= i && static_cast<dimn_t>(i) < m_size);
        return p_data[i];
    }

    template <typename I>
    constexpr enable_if_t<is_integral<I>::value, T&> operator[](I i
    ) const noexcept
    {
        RPY_DBG_ASSERT(0 <= i && static_cast<dimn_t>(i) < m_size);
        return p_data[i];
    }

    RPY_NO_DISCARD
    constexpr bool empty() const noexcept
    {
        return p_data == nullptr || m_size == 0;
    }

    RPY_NO_DISCARD
    constexpr std::size_t size() const noexcept { return m_size; }

    RPY_NO_DISCARD
    constexpr T* begin() noexcept { return p_data; }
    RPY_NO_DISCARD
    constexpr T* end() noexcept { return p_data + m_size; }
    RPY_NO_DISCARD
    constexpr const T* begin() const { return p_data; }
    RPY_NO_DISCARD
    constexpr const T* end() const { return p_data + m_size; }

    RPY_NO_DISCARD
    constexpr std::reverse_iterator<T*> rbegin() noexcept
    {
        return std::reverse_iterator<T*>(p_data + m_size);
    }
    RPY_NO_DISCARD
    constexpr std::reverse_iterator<T*> rend() noexcept
    {
        return std::reverse_iterator<T*>(p_data);
    }
    RPY_NO_DISCARD
    constexpr std::reverse_iterator<const T*> rbegin() const noexcept
    {
        return std::reverse_iterator<const T*>(p_data + m_size);
    }
    RPY_NO_DISCARD
    constexpr std::reverse_iterator<const T*> rend() const noexcept
    {
        return std::reverse_iterator<const T*>(p_data);
    }

    RPY_NO_DISCARD
    operator std::vector<T>() const
    {
        std::vector<T> result;
        result.reserve(m_size);
        for (dimn_t i = 0; i < m_size; ++i) { result.push_back(p_data[i]); }
        return result;
    }
};
}// namespace rpy

#endif// ROUGHPY_CORE_SLICE_H_
