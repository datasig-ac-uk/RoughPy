#ifndef ROUGHPY_COMPUTE__SRC_OBJECT_ARRAYS_OBJECT_ARRAY_ITERATOR_HPP
#define ROUGHPY_COMPUTE__SRC_OBJECT_ARRAYS_OBJECT_ARRAY_ITERATOR_HPP

#include "py_headers.h"

#include "py_obj_handle.hpp"

namespace rpy::compute {

template <typename Ref>
class ObjectArrayIterator
{
    PyObject** ptr_;

public:
    using value_type = Ref;
    using reference = Ref;
    using difference_type = Py_ssize_t;
    using pointer = PyObject**;

    using iterator_category = std::random_access_iterator_tag;

    constexpr explicit ObjectArrayIterator(PyObject** ptr) : ptr_(ptr) {}

    constexpr ObjectArrayIterator* operator++() noexcept
    {
        ++ptr_;
        return *this;
    }

    constexpr ObjectArrayIterator* operator++(int) noexcept
    {
        ObjectArrayIterator* result(*this);
        ++ptr_;
        return result;
    }

    constexpr ObjectArrayIterator* operator--() noexcept
    {
        --ptr_;
        return *this;
    }

    constexpr ObjectArrayIterator* operator--(int) noexcept
    {
        ObjectArrayIterator* result(*this);
        --ptr_;
        return result;
    }

    constexpr ObjectArrayIterator& operator+=(const difference_type n) noexcept
    {
        ptr_ += n;
        return *this;
    }

    constexpr ObjectArrayIterator& operator-=(const difference_type n) noexcept
    {
        ptr_ -= n;
        return *this;
    }

    constexpr reference operator*() const noexcept { return reference(ptr_); }

    constexpr reference operator[](const difference_type n) const noexcept
    {
        return reference(ptr_ + n);
    }

    friend constexpr ObjectArrayIterator
    operator+(const ObjectArrayIterator& lhs, difference_type rhs) noexcept
    {
        return ObjectArrayIterator(lhs.ptr_ + rhs);
    }

    friend constexpr difference_type operator-(
            const ObjectArrayIterator& lhs,
            const ObjectArrayIterator& rhs
    ) noexcept
    {
        return static_cast<difference_type>(lhs.ptr_ - rhs.ptr_);
    }

    friend constexpr bool operator==(
            const ObjectArrayIterator& lhs,
            const ObjectArrayIterator& rhs
    ) noexcept
    {
        return lhs.ptr_ == rhs.ptr_;
    }
    friend constexpr bool operator!=(
            const ObjectArrayIterator& lhs,
            const ObjectArrayIterator& rhs
    ) noexcept
    {
        return lhs.ptr_ != rhs.ptr_;
    }
    friend constexpr bool operator<(
            const ObjectArrayIterator& lhs,
            const ObjectArrayIterator& rhs
    ) noexcept
    {
        return lhs.ptr_ < rhs.ptr_;
    }
    friend constexpr bool operator<=(
            const ObjectArrayIterator& lhs,
            const ObjectArrayIterator& rhs
    ) noexcept
    {
        return lhs.ptr_ <= rhs.ptr_;
    }
    friend constexpr bool operator>(
            const ObjectArrayIterator& lhs,
            const ObjectArrayIterator& rhs
    ) noexcept
    {
        return lhs.ptr_ > rhs.ptr_;
    }
    friend constexpr bool operator>=(
            const ObjectArrayIterator& lhs,
            const ObjectArrayIterator& rhs
    ) noexcept
    {
        return lhs.ptr_ >= rhs.ptr_;
    }
};

using MutableObjectArrayIterator = ObjectArrayIterator<MutableObjectRef>;
using ConstObjectArrayIterator = ObjectArrayIterator<ObjectRef>;

}// namespace rpy::compute

#endif// ROUGHPY_COMPUTE__SRC_OBJECT_ARRAYS_OBJECT_ARRAY_ITERATOR_HPP
