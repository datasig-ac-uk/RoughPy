//
// Created by sam on 3/18/24.
//

#ifndef ROUGHPY_KEY_ARRAY_H
#define ROUGHPY_KEY_ARRAY_H

#include "basis_key.h"

#include <roughpy/core/macros.h>
#include <roughpy/core/ranges.h>
#include <roughpy/core/types.h>

#include <roughpy/scalars/devices/buffer.h>

namespace rpy {
namespace algebra {

namespace dtl {

class KeyArrayIterator
{
    const BasisKey* p_current;

public:
    using reference = const BasisKey&;
    using pointer = const BasisKey*;

    explicit KeyArrayIterator(pointer val) : p_current(val) {}

    KeyArrayIterator& operator++()
    {
        ++p_current;
        return *this;
    }
    const KeyArrayIterator operator++(int)
    {
        KeyArrayIterator prev(*this);
        this->operator++();
        return prev;
    }

    reference operator*()
    {
        RPY_DBG_ASSERT(p_current != nullptr);
        return *p_current;
    }
    pointer operator->()
    {
        RPY_DBG_ASSERT(p_current != nullptr);
        return p_current;
    }

    bool operator==(const KeyArrayIterator& other) const
    {
        return p_current == other.p_current;
    }
    bool operator!=(const KeyArrayIterator& other) const
    {
        return p_current != other.p_current;
    }
};

class KeyArrayRange
{
    devices::Buffer m_mapped_buffer;
    const BasisKey* p_begin;
    const BasisKey* p_end;

public:
    using reference = const BasisKey&;
    using pointer = const BasisKey*;
    using iterator = KeyArrayIterator;
    using const_iterator = iterator;


    explicit KeyArrayRange(devices::Buffer&& mapped) noexcept
            : m_mapped_buffer(std::move(mapped))
    {
        auto slice = m_mapped_buffer.as_slice<BasisKey>();
        p_begin = slice.begin();
        p_end = slice.end();
    }


    RPY_NO_DISCARD const_iterator begin() const noexcept
    {
        return KeyArrayIterator(p_begin);
    }
    RPY_NO_DISCARD const_iterator end() const noexcept
    {
        return KeyArrayIterator(p_end);
    }
};

}// namespace dtl

class ROUGHPY_ALGEBRA_EXPORT KeyArray
{
    devices::Buffer m_buffer;

public:
    using value_type = BasisKey;
    using iterator = dtl::KeyArrayIterator;
    using const_iterator = dtl::KeyArrayIterator;
    using reference = BasisKey&;
    using const_reference = const BasisKey&;

    KeyArray();
    KeyArray(const KeyArray&);
    KeyArray(KeyArray&&) noexcept;

    explicit KeyArray(devices::Buffer&& data) : m_buffer(std::move(data))
    {
        RPY_CHECK(m_buffer.type_info().code == devices::TypeCode::KeyType);
    }

    explicit KeyArray(Slice<BasisKey> keys);

    explicit KeyArray(dimn_t size);

    ~KeyArray();

    RPY_NO_DISCARD bool empty() const noexcept { return m_buffer.empty(); }
    RPY_NO_DISCARD dimn_t size() const noexcept { return m_buffer.size(); }

    KeyArray& operator=(const KeyArray&);
    KeyArray& operator=(KeyArray&&) noexcept;

    dtl::KeyArrayRange as_range(dimn_t offset = 0, dimn_t end_offset = 0) const
    {
        RPY_CHECK(offset <= m_buffer.size() && end_offset <= m_buffer.size());
        if (end_offset == 0) { end_offset = m_buffer.size(); }
        RPY_CHECK(end_offset >= offset);
        return dtl::KeyArrayRange(m_buffer.map(offset, end_offset - offset));
    }

    BasisKey operator[](dimn_t index) const;

    BasisKey& operator[](dimn_t index);

    devices::Buffer& mut_buffer() noexcept { return m_buffer; }
    const devices::Buffer& buffer() const noexcept { return m_buffer; }

    RPY_NO_DISCARD devices::Device device() const noexcept
    {
        return m_buffer.device();
    }

    RPY_NO_DISCARD Slice<const BasisKey> as_slice() const
    {
        return m_buffer.as_slice<BasisKey>();
    }

    RPY_NO_DISCARD Slice<BasisKey> as_mut_slice()
    {
        return m_buffer.as_mut_slice<BasisKey>();
    }

    RPY_NO_DISCARD KeyArray view() const { return KeyArray(m_buffer.map()); }
    RPY_NO_DISCARD KeyArray mut_view() { return KeyArray(m_buffer.map()); }

    template <typename ViewFn>
    friend constexpr auto
    operator|(const KeyArray& array, views::view_closure<ViewFn>& view)
            -> decltype(array.as_range() | view)
    {
        return array.as_range() | view;
    }
};

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_KEY_ARRAY_H
