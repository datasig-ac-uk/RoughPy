//
// Created by sam on 3/18/24.
//

#ifndef ROUGHPY_KEY_ARRAY_H
#define ROUGHPY_KEY_ARRAY_H

#include "basis_key.h"

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include <roughpy/scalars/devices/buffer.h>

namespace rpy {
namespace algebra {

namespace dtl {

class ROUGHPY_ALGEBRA_EXPORT KeyArrayIterator
{

public:
    using reference = void;
    using pointer = void*;

    keyArrayIterator& operator++();
    KeyArrayIterator& operator++(int) const;

    reference operator*();
    pointer operator->();

    bool operator==(const KeyArrayIterator&) const;
    bool operator!=(const KeyArrayIterator&) const;
};

}// namespace dtl

class KeyArray
{
    devices::Buffer m_buffer;

public:


    KeyArray();
    KeyArray(const KeyArray&);
    KeyArray(KeyArray&&) noexcept;

    explicit KeyArray(devices::Buffer&& data) : m_buffer(std::move(data))
    {
        RPY_CHECK(m_buffer.type_info().code == devices::TypeCode::KeyType);
    }

    explicit KeyArray(Slice<BasisKey> keys);

    ~KeyArray();

    RPY_NO_DISCARD bool empty() const noexcept { return m_buffer.empty(); }
    RPY_NO_DISCARD dimn_t size() const noexcept { return m_buffer.size(); }

    KeyArray& operator=(const KeyArray&);
    KeyArray& operator=(KeyArray&&) noexcept;

    BasisKey operator[](dimn_t index) const;

    BasisKey& operator[](dimn_t index);

    devices::Buffer& mut_buffer() noexcept { return m_buffer; }
    const devices::Buffer& buffer() const noexcept { return m_buffer; }

    RPY_NO_DISCARD
    devices::Device device() const noexcept { return m_buffer.device(); }

    RPY_NO_DISCARD Slice<const BasisKey> as_slice() const {
        return m_buffer.as_slice<BasisKey>();
    }

    RPY_NO_DISCARD Slice<BasisKey> as_mut_slice() {
        return m_buffer.as_mut_slice<BasisKey>();
    }

    RPY_NO_DISCARD KeyArray view() const { return KeyArray(m_buffer.map()); }
    RPY_NO_DISCARD KeyArray mut_view() { return KeyArray(m_buffer.map()); }


};

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_KEY_ARRAY_H
