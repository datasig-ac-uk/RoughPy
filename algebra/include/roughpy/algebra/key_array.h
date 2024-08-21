//
// Created by sam on 3/18/24.
//

#ifndef ROUGHPY_KEY_ARRAY_H
#define ROUGHPY_KEY_ARRAY_H

#include "algebra_fwd.h"

#include <roughpy/core/macros.h>
#include <roughpy/core/ranges.h>
#include <roughpy/core/types.h>

#include <roughpy/devices/buffer.h>

#include "basis_keys.h"

namespace rpy {
namespace algebra {

namespace dtl {

class KeyArrayIterator
{
    const BasisKey* p_current = nullptr;

public:
    using value_type = const BasisKey;
    using reference = const BasisKey&;
    using pointer = const BasisKey*;
    using size_type = dimn_t;
    using difference_type = idimn_t;

    using iterator_category = std::random_access_iterator_tag;

    KeyArrayIterator() = default;
    KeyArrayIterator(const KeyArrayIterator&) = default;
    KeyArrayIterator(KeyArrayIterator&&) noexcept = default;

    constexpr explicit KeyArrayIterator(pointer val) : p_current(val) {}

    KeyArrayIterator& operator=(const KeyArrayIterator&) = default;
    KeyArrayIterator& operator=(KeyArrayIterator&&) noexcept = default;

    constexpr KeyArrayIterator& operator++() noexcept
    {
        ++p_current;
        return *this;
    }

    constexpr const KeyArrayIterator operator++(int) noexcept
    {
        KeyArrayIterator prev(*this);
        this->operator++();
        return prev;
    }

    constexpr KeyArrayIterator& operator--() noexcept
    {
        --p_current;
        return *this;
    }

    constexpr const KeyArrayIterator operator--(int) noexcept
    {
        KeyArrayIterator prev(*this);
        --prev;
        return prev;
    }

    constexpr KeyArrayIterator& operator+=(difference_type n) noexcept
    {
        p_current += n;
        return *this;
    }

    constexpr KeyArrayIterator& operator-=(difference_type n) noexcept
    {
        p_current -= n;
        return *this;
    }

    constexpr reference operator*()
    {
        RPY_DBG_ASSERT(p_current != nullptr);
        return *p_current;
    }
    constexpr pointer operator->()
    {
        RPY_DBG_ASSERT(p_current != nullptr);
        return p_current;
    }

    constexpr reference operator[](difference_type n) noexcept
    {
        return p_current[n];
    }

    constexpr bool operator==(const KeyArrayIterator& other) const
    {
        return p_current == other.p_current;
    }
    constexpr bool operator!=(const KeyArrayIterator& other) const
    {
        return p_current != other.p_current;
    }
};

constexpr KeyArrayIterator
operator+(const KeyArrayIterator& it, idimn_t n) noexcept
{
    KeyArrayIterator next(it);
    next += n;
    return next;
}

constexpr KeyArrayIterator
operator+(idimn_t n, const KeyArrayIterator& it) noexcept
{
    KeyArrayIterator next(it);
    next += n;
    return next;
}

constexpr KeyArrayIterator
operator-(const KeyArrayIterator& it, idimn_t n) noexcept
{
    KeyArrayIterator prev(it);
    prev -= n;
    return prev;
}

class KeyArrayRange
{
    devices::Buffer m_mapped_buffer{};

public:
    using reference = const BasisKey&;
    using pointer = const BasisKey*;
    using iterator = KeyArrayIterator;
    using const_iterator = iterator;

    explicit KeyArrayRange(devices::Buffer&& mapped) noexcept
        : m_mapped_buffer(std::move(mapped))
    {}

    RPY_NO_DISCARD const_iterator begin() const noexcept { return {}; }
    RPY_NO_DISCARD const_iterator end() const noexcept { return {}; }

    RPY_NO_DISCARD iterator begin() noexcept { return KeyArrayIterator(); }
    RPY_NO_DISCARD iterator end() noexcept { return KeyArrayIterator(); }
};

}// namespace dtl

/**
 * @class KeyArray
 * @brief Represents an array of BasisKey objects.
 *
 * The KeyArray class provides a container for storing and manipulating an array
 * of BasisKey objects. It internally uses a devices::Buffer to store the data.
 * The class provides various member functions and operators for accessing and
 * manipulating the data in the KeyArray.
 *
 * @tparam BasisKey The type of the elements stored in the KeyArray.
 */
class ROUGHPY_ALGEBRA_EXPORT KeyArray : public devices::Buffer
{
public:
    using value_type = BasisKey;
    using iterator = dtl::KeyArrayIterator;
    using const_iterator = dtl::KeyArrayIterator;
    using reference = BasisKeyRef;
    using const_reference = BasisKeyCRef;
    using Buffer::Buffer;

    explicit KeyArray(Buffer&& buffer) noexcept : Buffer(std::move(buffer)) {}

    KeyArray& operator=(const KeyArray&);
    KeyArray& operator=(KeyArray&&) noexcept;

    RPY_NO_DISCARD bool is_const() const noexcept
    {
        return mode() == devices::BufferMode::Read;
    }

    BasisKey operator[](dimn_t index) const;

    BasisKey& operator[](dimn_t index);

    KeyArray operator[](SliceIndex index);
    KeyArray operator[](SliceIndex index) const;

    devices::Buffer& mut_buffer() noexcept
    {
        RPY_CHECK(!is_const());
        return *this;
    }
    const devices::Buffer& buffer() const noexcept { return *this; }

    /**
     * @brief Returns a constant slice of BasisKey objects from the KeyArray.
     *
     * The as_slice method returns a constant slice of BasisKey objects from the
     * KeyArray. It internally calls the as_slice method of the devices::Buffer
     * class, passing the BasisKey type as a template parameter. This method
     * only returns a constant view of the data, allowing read-only access to
     * the elements.
     *
     * @return A constant Slice object representing the slice of BasisKey
     * objects.
     * @note This method does not throw any exceptions.
     * @see devices::Buffer::as_slice()
     */
    // RPY_NO_DISCARD Slice<const BasisKey> as_slice() const
    // {
    //     return m_buffer.as_value_slice();
    // }

    /**
     * @brief Returns a mutable slice of BasisKey objects from the KeyArray.
     *
     * The as_mut_slice method returns a mutable slice of BasisKey objects from
     * the KeyArray. It internally calls the as_mut_slice method of the
     * devices::Buffer class, passing the BasisKey type as a template parameter.
     * This method provides a mutable view of the data, allowing read and write
     * access to the elements.
     *
     * @return A mutable Slice object representing the slice of BasisKey
     * objects.
     * @note This method does not throw any exceptions.
     * @see devices::Buffer::as_mut_slice()
     */
    // RPY_NO_DISCARD Slice<BasisKey> as_mut_slice()
    // {
    //     return m_buffer.as_mut_value_slice();
    // }

    /**
     * @brief Returns a KeyArray object representing a view of the data in the
     * buffer.
     *
     * The view() method returns a KeyArray object which represents a view of
     * the data stored in the buffer. The view is created by mapping the buffer
     * using the `map()` method of the buffer object. The KeyArray object
     * returned by this method can be used to access and manipulate the data in
     * the buffer.
     *
     * @return A KeyArray object representing a view of the data in the buffer.
     *
     * @see KeyArray
     * @see Buffer::map()
     */
    RPY_NO_DISCARD KeyArray view() const { return KeyArray(map()); }

    /**
     * @brief Creates a mutable view of the KeyArray object.
     *
     * The `mut_view()` method returns a new KeyArray object that provides a
     * mutable view of the data stored in the current KeyArray object. The
     * method internally uses the `map()` function of the m_buffer member
     * variable to create the mutable view.
     *
     * The mutable view allows modifying the elements of the KeyArray object
     * without making a copy of the underlying data. Any changes made to the
     * mutable view will directly affect the data stored in the original
     * KeyArray object.
     *
     * @note It's important to make sure that the mutable view is not used after
     * the original KeyArray object is destroyed, as it will lead to undefined
     * behavior.
     *
     * @return A new KeyArray object that provides a mutable view of the data
     * stored in the original KeyArray object.
     */
    RPY_NO_DISCARD KeyArray mut_view() { return KeyArray(map()); }

    /**
     * @brief Copies the KeyArray to the specified device.
     *
     * The to_device method creates a new KeyArray object by copying the
     * elements of the current KeyArray to the specified device. It internally
     * allocates a devices::Buffer on the specified device and copies the data
     * from the original devices::Buffer using the to_device method.
     *
     * @tparam BasisKey The type of the elements stored in the KeyArray.
     * @param device The target device to copy the KeyArray to.
     * @return A new KeyArray object containing the copied data on the specified
     * device.
     */
    RPY_NO_DISCARD KeyArray to_device(devices::Device device) const;

    // template <typename ViewFn>
    // friend constexpr auto
    // operator|(const KeyArray& array, views::view_closure<ViewFn>& view)
    //         -> decltype(array.as_range() | view)
    // {
    //     return array.as_range() | view;
    // }
};

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_KEY_ARRAY_H
