#ifndef ROUGHPY_COMPUTE__SRC_OBJECT_ARRAYS_OBJECT_ARRAY_ITERATOR_HPP
#define ROUGHPY_COMPUTE__SRC_OBJECT_ARRAYS_OBJECT_ARRAY_ITERATOR_HPP

#include "py_headers.h"


#include <iterator>
#include <type_traits>

#include "py_obj_handle.hpp"

namespace rpy::compute {

template <typename Ref, typename Iterator>
class ObjectArrayIterator
{
    using Traits = std::iterator_traits<Iterator>;
    static_assert(
            std::is_base_of_v<
                    std::random_access_iterator_tag,
                    typename Traits::iterator_category>,
            "base iterator must be random access"
    );


    Iterator base_;

public:
    using value_type = typename Traits::value_type;
    using reference = Ref;
    using difference_type = typename Traits::difference_type;

    using iterator_category = std::random_access_iterator_tag;

    constexpr explicit ObjectArrayIterator(Iterator ptr) : base_(ptr) {}

    constexpr ObjectArrayIterator* operator++() noexcept
    {
        ++base_;
        return *this;
    }

    constexpr ObjectArrayIterator* operator++(int) noexcept
    {
        ObjectArrayIterator* result(*this);
        ++base_;
        return result;
    }

    constexpr ObjectArrayIterator* operator--() noexcept
    {
        --base_;
        return *this;
    }

    constexpr ObjectArrayIterator* operator--(int) noexcept
    {
        ObjectArrayIterator* result(*this);
        --base_;
        return result;
    }

    constexpr ObjectArrayIterator& operator+=(const difference_type n) noexcept
    {
        base_ += n;
        return *this;
    }

    constexpr ObjectArrayIterator& operator-=(const difference_type n) noexcept
    {
        base_ -= n;
        return *this;
    }

    constexpr reference operator*() const noexcept { return reference(std::addressof(*base_)); }

    constexpr reference operator[](const difference_type n) const noexcept
    {
        return reference(std::addressof(*(base_ + n)));
    }

    friend constexpr ObjectArrayIterator
    operator+(const ObjectArrayIterator& lhs, difference_type rhs) noexcept
    {
        return ObjectArrayIterator(lhs.base_ + rhs);
    }

    friend constexpr difference_type operator-(
            const ObjectArrayIterator& lhs,
            const ObjectArrayIterator& rhs
    ) noexcept
    {
        return static_cast<difference_type>(lhs.base_ - rhs.base_);
    }

    friend constexpr bool operator==(
            const ObjectArrayIterator& lhs,
            const ObjectArrayIterator& rhs
    ) noexcept
    {
        return lhs.base_ == rhs.base_;
    }
    friend constexpr bool operator!=(
            const ObjectArrayIterator& lhs,
            const ObjectArrayIterator& rhs
    ) noexcept
    {
        return lhs.base_ != rhs.base_;
    }
    friend constexpr bool operator<(
            const ObjectArrayIterator& lhs,
            const ObjectArrayIterator& rhs
    ) noexcept
    {
        return lhs.base_ < rhs.base_;
    }
    friend constexpr bool operator<=(
            const ObjectArrayIterator& lhs,
            const ObjectArrayIterator& rhs
    ) noexcept
    {
        return lhs.base_ <= rhs.base_;
    }
    friend constexpr bool operator>(
            const ObjectArrayIterator& lhs,
            const ObjectArrayIterator& rhs
    ) noexcept
    {
        return lhs.base_ > rhs.base_;
    }
    friend constexpr bool operator>=(
            const ObjectArrayIterator& lhs,
            const ObjectArrayIterator& rhs
    ) noexcept
    {
        return lhs.base_ >= rhs.base_;
    }
};

template <typename Iter>
using MutableObjectArrayIterator = ObjectArrayIterator<MutableObjectRef, Iter>;

template <typename Iter>
using ConstObjectArrayIterator = ObjectArrayIterator<ObjectRef, Iter>;

}// namespace rpy::compute

#endif// ROUGHPY_COMPUTE__SRC_OBJECT_ARRAYS_OBJECT_ARRAY_ITERATOR_HPP
