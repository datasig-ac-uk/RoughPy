//
// Created by sam on 25/08/24.
//

#ifndef TO_LETTER_ITERATOR_H
#define TO_LETTER_ITERATOR_H

#include <roughpy/core/types.h>

#include <iterator>

namespace rpy {
namespace algebra {
namespace dtl {

class LetterPtrProxy
{
    let_t value;

public:
    constexpr LetterPtrProxy(let_t value) : value(value) {}

    constexpr const let_t* operator->() const noexcept { return &value; }
    constexpr const let_t& operator*() const noexcept { return value; }
};

class ToLetterIterator
{
    dimn_t m_index = 0;
    deg_t m_width = 0;
    deg_t m_degree = 0;

public:
    using value_type = let_t;
    using difference_type = idimn_t;
    using reference = value_type;
    using pointer = LetterPtrProxy;
    using iterator_category = std::forward_iterator_tag;

    constexpr ToLetterIterator() = default;

    constexpr ToLetterIterator(dimn_t index, deg_t width, deg_t degree);

    constexpr reference operator*() const noexcept
    {
        return static_cast<value_type>(m_index % m_width);
    }

    constexpr pointer operator->() const noexcept
    {
        return {this->operator*()};
    }

    constexpr ToLetterIterator& operator++() noexcept
    {
        m_index /= m_width;
        return *this;
    }

    constexpr const ToLetterIterator operator++(int) noexcept
    {
        ToLetterIterator tmp(*this);
        operator++();
        return tmp;
    }

    friend constexpr bool operator==(
            const ToLetterIterator& lhs,
            const ToLetterIterator& rhs
    ) noexcept
    {
        return lhs.m_degree == 0;
    }

    friend constexpr bool operator!=(
            const ToLetterIterator& lhs,
            const ToLetterIterator& rhs
    ) noexcept
    {
        return lhs.m_degree != 0;
    }
};


class ToLetterRange
{
    ToLetterIterator m_begin;
    ToLetterIterator m_end {};

public:

    using iterator = ToLetterIterator;
    using const_iterator = ToLetterIterator;

    ToLetterRange(dimn_t index, deg_t width, deg_t degree)
        : m_begin(index, width, degree)
    {}


    constexpr const_iterator begin() const noexcept { return m_begin; }
    constexpr const_iterator end() const noexcept { return m_end; }

};

}// namespace dtl
}// namespace algebra
}// namespace rpy

#endif// TO_LETTER_ITERATOR_H
