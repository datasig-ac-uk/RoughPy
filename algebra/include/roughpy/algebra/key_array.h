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

    explicit KeyArray(devices::Buffer&& data) : m_buffer(std::move(data)) {}

    explicit KeyArray(Slice<BasisKey> keys);

    explicit KeyArray(dimn_t size);

    explicit KeyArray(devices::Device device, dimn_t size);

    ~KeyArray();

    /**
     * @brief Checks if the buffer is empty.
     *
     * The empty method checks whether the buffer is empty or not. It returns a
     * boolean value indicating whether the buffer is empty.
     *
     * @return A boolean value. True if the buffer is empty, false otherwise.
     * @note This method does not throw any exceptions and is guaranteed to be
     * noexcept.
     */
    RPY_NO_DISCARD bool empty() const noexcept { return m_buffer.empty(); }

    /**
     * @brief Gets the size of the buffer.
     *
     * The size method returns the size of the buffer. It returns a dimn_t value
     * indicating the size of the buffer in elements.
     *
     * @return The size of the buffer.
     * @note This method does not throw any exceptions and is guaranteed to be
     * noexcept.
     */
    RPY_NO_DISCARD dimn_t size() const noexcept { return m_buffer.size(); }

    KeyArray& operator=(const KeyArray&);
    KeyArray& operator=(KeyArray&&) noexcept;

    /**
     * @brief Returns a range of BasisKey objects in the KeyArray.
     *
     * The as_range method returns a dtl::KeyArrayRange object representing a
     * range of BasisKey objects in the KeyArray. The range starts at the given
     * offset and ends at the given end_offset (or the end of the KeyArray if
     * end_offset is not specified). The method checks that the offsets are
     * within the bounds of the KeyArray buffer before creating the range. If
     * the offsets are invalid, an exception will be thrown.
     *
     * @param offset The starting offset of the range. Defaults to 0.
     * @param end_offset The ending offset of the range. Defaults to 0, which
     * means it will be set to the size of the KeyArray buffer.
     * @return A dtl::KeyArrayRange object representing the range of BasisKey
     * objects.
     * @throws std::runtime_error if the offset or end_offset is out of bounds.
     */
    dtl::KeyArrayRange as_range(dimn_t offset = 0, dimn_t end_offset = 0) const
    {
        RPY_CHECK(offset <= m_buffer.size() && end_offset <= m_buffer.size());
        if (end_offset == 0) { end_offset = m_buffer.size(); }
        RPY_CHECK(end_offset >= offset);
        return dtl::KeyArrayRange(m_buffer.map(offset, end_offset - offset));
    }

    BasisKey operator[](dimn_t index) const;

    BasisKey& operator[](dimn_t index);

    KeyArray operator[](SliceIndex index);
    KeyArray operator[](SliceIndex index) const;

    devices::Buffer& mut_buffer() noexcept { return m_buffer; }
    const devices::Buffer& buffer() const noexcept { return m_buffer; }

    /**
     * @brief Returns the underlying device associated with this device.
     *
     * The device() method returns the underlying devices::Device object
     * associated with this device. This method is const and noexcept.
     *
     * @return The underlying devices::Device object associated with this
     * device.
     */
    RPY_NO_DISCARD devices::Device device() const noexcept
    {
        return m_buffer.device();
    }

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
    RPY_NO_DISCARD KeyArray view() const { return KeyArray(m_buffer.map()); }

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
    RPY_NO_DISCARD KeyArray mut_view() { return KeyArray(m_buffer.map()); }

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
