// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
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

#include "container/vector.h"
#include "ranges.h"
#include "traits.h"
#include "types.h"

#include <iterator>
#include <memory>
#include <utility>

namespace rpy {

namespace dtl {

/*
 * This SliceIterator type is based heavily on the implementation of
 * the __bounded_iter from LLVM libstdcxx and the span_iterator from the
 * Microsoft GSL library. It implements bounds checks on iterator access to
 * prevent accesses outside of the given range.
 */

template <typename ElementType>
class SliceIterator
{
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = remove_cv_t<ElementType>;
    using difference_type = idimn_t;
    using pointer = ElementType*;
    using reference = ElementType&;

private:
    pointer p_begin;
    pointer p_end;
    pointer p_current;

public:
    constexpr SliceIterator() noexcept = default;
    constexpr SliceIterator(const SliceIterator& other) noexcept = default;
    constexpr SliceIterator(SliceIterator&& other) noexcept = default;

    constexpr
    SliceIterator(pointer begin, pointer end, pointer current) noexcept
        : p_begin(begin),
          p_end(end),
          p_current(current)
    {}

    constexpr SliceIterator& operator=(const SliceIterator& other) noexcept
            = default;
    constexpr SliceIterator& operator=(SliceIterator&& other) noexcept
            = default;

    constexpr operator SliceIterator<const value_type>() const noexcept
    {
        return {p_begin, p_end, p_current};
    }

    RPY_NO_DISCARD constexpr reference operator*() const noexcept
    {
        RPY_DBG_ASSERT(p_begin != nullptr && p_end != nullptr);
        RPY_DBG_ASSERT(p_begin <= p_current && p_current < p_end);
        return *p_current;
    }

    RPY_NO_DISCARD constexpr reference operator[](difference_type index) const
    {
        return *(*this + index);
    }

    RPY_NO_DISCARD constexpr pointer operator->() const noexcept
    {
        RPY_DBG_ASSERT(p_begin != nullptr && p_end != nullptr);
        RPY_DBG_ASSERT(p_begin <= p_current && p_current < p_end);
        return p_current;
    }

    constexpr SliceIterator& operator++() noexcept
    {
        RPY_DBG_ASSERT(p_current != nullptr && p_end != nullptr);
        RPY_DBG_ASSERT(p_current < p_end);
        ++p_current;
        return *this;
    }

    RPY_NO_DISCARD constexpr SliceIterator operator++(int) noexcept
    {
        SliceIterator prev(*this);
        ++(*this);
        return prev;
    }

    constexpr SliceIterator& operator--() noexcept
    {
        RPY_DBG_ASSERT(p_begin != nullptr && p_end != nullptr);
        RPY_DBG_ASSERT(p_begin < p_current);
        ++p_current;
        return *this;
    }

    RPY_NO_DISCARD constexpr SliceIterator operator--(int) noexcept
    {
        SliceIterator next(*this);
        --(*this);
        return next;
    }

    constexpr SliceIterator& operator+=(difference_type n) noexcept
    {
        RPY_DBG_ASSERT(
                n == 0
                || (p_begin != nullptr && p_current != nullptr
                    && p_end != nullptr)
        );
        RPY_DBG_ASSERT(
                n > 0 ? p_end - p_current >= n
                      : (n == 0 || (p_current - p_begin) >= -n)
        );

        p_current += n;
        return *this;
    }

    RPY_NO_DISCARD constexpr SliceIterator operator+(difference_type n
    ) const noexcept
    {
        SliceIterator advanced(*this);
        advanced += n;
        return advanced;
    }

    RPY_NO_DISCARD friend constexpr SliceIterator
    operator+(difference_type n, const SliceIterator& rhs) noexcept
    {
        return rhs + n;
    }

    constexpr SliceIterator& operator-=(difference_type n) noexcept
    {
        RPY_DBG_ASSERT(
                n == 0
                || (p_begin != nullptr && p_current != nullptr
                    && p_end != nullptr)
        );
        RPY_DBG_ASSERT(
                n < 0 ? p_end - p_current >= -n
                      : (n == 0 || (p_current - p_begin) >= n)
        );

        p_current -= n;
        return *this;
    }

    RPY_NO_DISCARD constexpr SliceIterator operator-(difference_type n
    ) const noexcept
    {
        SliceIterator advanced(*this);
        advanced -= n;
        return advanced;
    }

    template <
            typename EltType,
            typename = enable_if_t<is_same_v<remove_cv_t<EltType>, value_type>>>
    RPY_NO_DISCARD constexpr difference_type
    operator-(const SliceIterator<EltType>& rhs) const noexcept
    {
        RPY_DBG_ASSERT(p_current == rhs.p_current && p_end == rhs.p_end);
        return p_current - rhs.p_current;
    }

    template <
            typename EltType,
            typename = enable_if_t<is_same_v<remove_cv_t<EltType>, value_type>>>
    RPY_NO_DISCARD constexpr bool operator==(const SliceIterator<EltType>& rhs
    ) const noexcept
    {
        RPY_DBG_ASSERT(p_current == rhs.p_current && p_end == rhs.p_end);
        return p_current == rhs.p_current;
    }

    template <
            typename EltType,
            typename = enable_if_t<is_same_v<remove_cv_t<EltType>, value_type>>>
    RPY_NO_DISCARD constexpr bool operator!=(const SliceIterator<EltType>& rhs
    ) const noexcept
    {
        RPY_DBG_ASSERT(p_current == rhs.p_current && p_end == rhs.p_end);
        return p_current != rhs.p_current;
    }

    template <
            typename EltType,
            typename = enable_if_t<is_same_v<remove_cv_t<EltType>, value_type>>>
    RPY_NO_DISCARD constexpr bool operator<(const SliceIterator<EltType>& rhs
    ) const noexcept
    {
        RPY_DBG_ASSERT(p_current == rhs.p_current && p_end == rhs.p_end);
        return p_current < rhs.p_current;
    }

    template <
            typename EltType,
            typename = enable_if_t<is_same_v<remove_cv_t<EltType>, value_type>>>
    RPY_NO_DISCARD constexpr bool operator<=(const SliceIterator<EltType>& rhs
    ) const noexcept
    {
        RPY_DBG_ASSERT(p_current == rhs.p_current && p_end == rhs.p_end);
        return p_current <= rhs.p_current;
    }

    template <
            typename EltType,
            typename = enable_if_t<is_same_v<remove_cv_t<EltType>, value_type>>>
    RPY_NO_DISCARD constexpr bool operator>(const SliceIterator<EltType>& rhs
    ) const noexcept
    {
        RPY_DBG_ASSERT(p_current == rhs.p_current && p_end == rhs.p_end);
        return p_current > rhs.p_current;
    }

    template <
            typename EltType,
            typename = enable_if_t<is_same_v<remove_cv_t<EltType>, value_type>>>
    RPY_NO_DISCARD constexpr bool operator>=(const SliceIterator<EltType>& rhs
    ) const noexcept
    {
        RPY_DBG_ASSERT(p_current == rhs.p_current && p_end == rhs.p_end);
        return p_current >= rhs.p_current;
    }

private:
    template <typename Ptr>
    friend class std::pointer_traits;
};

}// namespace dtl

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
    dimn_t m_size = 0;

public:
    using element_type = T;
    using value_type = remove_cv_t<T>;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = dimn_t;
    using difference_type = idimn_t;

    // using iterator = T*;
    using iterator = dtl::SliceIterator<element_type>;
    // using const_iterator = dtl::SliceIterator<const value_type>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    // using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    constexpr Slice() noexcept = default;
    constexpr Slice(const Slice&) noexcept = default;
    constexpr Slice(Slice&&) noexcept = default;

    constexpr Slice(T& num) noexcept : p_data(&num), m_size(1) {}

    constexpr Slice(std::initializer_list<T> data) noexcept
        : p_data(data.data()),
          m_size(data.size())
    {}

    constexpr Slice(std::nullptr_t) noexcept : p_data(nullptr), m_size(0) {}

    template <
            typename Container,
            typename
            = enable_if_t<is_same_v<typename Container::value_type, T>>>
    constexpr Slice(Container& container) noexcept
        : p_data(container.data()),
          m_size(container.size())
    {}

    template <
            typename Container,
            typename = enable_if_t<
                    is_same_v<remove_cv_t<typename Container::value_type>, T>>>
    constexpr Slice(const Container& container) noexcept
        : p_data(container.data()),
          m_size(container.size())
    {}

    template <std::size_t N>
    constexpr Slice(T (&array)[N]) noexcept : p_data(array),
                                              m_size(N)
    {}

    constexpr Slice(T* ptr, size_type N) noexcept : p_data(ptr), m_size(N) {}

    constexpr operator Slice<add_const_t<T>>() const noexcept
    {
        return {p_data, m_size};
    }

    template <typename Container>
    enable_if_t<
            is_const_v<T>
                    && is_same_v<
                            remove_const_t<T>,
                            typename Container::value_type>,
            Slice>
    operator=(const Container& container) noexcept
    {
        p_data = container.data();
        m_size = container.size();
        return *this;
    }

    constexpr Slice& operator=(const Slice&) = default;
    constexpr Slice& operator=(Slice&&) noexcept = default;

    template <typename I>
    constexpr enable_if_t<is_integral_v<I>, T&> operator[](I i) noexcept
    {
        RPY_DBG_ASSERT(0 <= i && static_cast<dimn_t>(i) < m_size);
        return p_data[i];
    }

    template <typename I>
    constexpr enable_if_t<is_integral_v<I>, const T&> operator[](I i
    ) const noexcept
    {
        RPY_DBG_ASSERT(0 <= i && static_cast<dimn_t>(i) < m_size);
        return p_data[i];
    }

    RPY_NO_DISCARD constexpr bool empty() const noexcept
    {
        return p_data == nullptr || m_size == 0;
    }

    RPY_NO_DISCARD constexpr size_type size() const noexcept { return m_size; }

    RPY_NO_DISCARD constexpr T* data() const noexcept { return p_data; }

    RPY_NO_DISCARD constexpr iterator begin() const noexcept
    {
        return {p_data, p_data + m_size, p_data};
        // return p_data;
    }
    RPY_NO_DISCARD constexpr iterator end() const noexcept
    {
        return {p_data, p_data + m_size, p_data + m_size};
        // return p_data + m_size;
    }

    RPY_NO_DISCARD constexpr reverse_iterator rbegin() const noexcept
    {
        return reverse_iterator{end()};
    }
    RPY_NO_DISCARD constexpr reverse_iterator rend() const noexcept
    {
        return reverse_iterator{begin()};
    }

    RPY_NO_DISCARD operator containers::Vec<remove_const_t<T>>() const
    {
        containers::Vec<T> result;
        result.reserve(m_size);
        for (dimn_t i = 0; i < m_size; ++i) { result.push_back(p_data[i]); }
        return result;
    }
};

template <>
class Slice<void>
{
    void* p_data = nullptr;
    dimn_t m_size = 0;

public:
    constexpr Slice() = default;

    template <typename T>
    constexpr Slice(T& num) : p_data(&num),
                              m_size(1)
    {}

    constexpr Slice(std::nullptr_t) : p_data(nullptr), m_size(0) {}

    template <typename T>
    constexpr Slice(containers::Vec<T>& container)
        : p_data(container.data()),
          m_size(container.size())
    {}

    template <typename T, std::size_t N>
    constexpr Slice(T (&array)[N]) : p_data(array),
                                     m_size(N)
    {}

    template <typename T>
    constexpr Slice(T* ptr, std::size_t N) : p_data(ptr),
                                             m_size(N)
    {}

    RPY_NO_DISCARD constexpr bool empty() const noexcept
    {
        return p_data == nullptr || m_size == 0;
    }

    RPY_NO_DISCARD constexpr std::size_t size() const noexcept
    {
        return m_size;
    }

    RPY_NO_DISCARD constexpr void* data() const noexcept { return p_data; }
};

using CHECK_ME_begin
        = decltype(ranges::begin(std::declval<const Slice<float>&>()));
using CHECK_ME_end = decltype(ranges::end(std::declval<const Slice<float>&>()));

}// namespace rpy

namespace std {

template <typename EltType>
struct pointer_traits<::rpy::dtl::SliceIterator<EltType>> {
    using pointer = ::rpy::dtl::SliceIterator<EltType>;
    using element_type = EltType;
    using difference_type = ptrdiff_t;

    static constexpr element_type* to_address(const pointer i)
    {
        return i.p_current;
    }
};

}// namespace std

#endif// ROUGHPY_CORE_SLICE_H_
