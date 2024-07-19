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

#ifndef ROUGHPY_DEVICE_BUFFER_H_
#define ROUGHPY_DEVICE_BUFFER_H_

#include "core.h"
#include "device_object_base.h"
#include "type.h"

#include <roughpy/core/macros.h>
#include <roughpy/core/slice.h>
#include <roughpy/core/types.h>

#include "value.h"

namespace rpy {

template <>
class Slice<devices::Value>;

template <>
class Slice<const devices::Value>;

namespace devices {

/**
 * @class BufferInterface
 * @brief Interface for buffer objects.
 *
 * This interface provides methods for manipulating buffer objects. It is used
 * as a base for the `Buffer` class.
 */
class ROUGHPY_DEVICES_EXPORT BufferInterface : public dtl::InterfaceBase
{
    /*
     * I'd really like to have the memory owner pointer here, but I can't
     * because the Buffer class isn't defined until later. Instead, the
     * implementations will have to handle this by themselves.
     */
    TypePtr p_type;

public:
    using object_t = Buffer;

    explicit BufferInterface(TypePtr type) : p_type(std::move(type)) {}

    /**
     * @brief Get the content type of the object.
     *
     * This method retrieves the content type of the object.
     *
     * @return A pointer to the Type representing the content type of the
     * object. Returns nullptr if the object is null.
     */
    RPY_NO_DISCARD TypePtr type() const noexcept { return p_type; }

    /**
     * @brief Returns the mode of the buffer object.
     *
     * This is a virtual method that returns the mode of the buffer object. It
     * is a const method which means it does not modify the object.
     *
     * @return The mode of the buffer object.
     *
     * @see BufferMode
     */
    RPY_NO_DISCARD virtual BufferMode mode() const;
    /**
     * @brief Get the size of the buffer object.
     *
     * This method retrieves the size of the buffer object. It returns the
     * number of elements in the buffer.
     *
     * @return The size of the buffer object.
     */
    RPY_NO_DISCARD virtual dimn_t size() const;
    /**
     * @brief Returns the total number of bytes in the buffer.
     *
     * This method calculates the total number of bytes in the buffer. It
     * considers the size of the buffer and the number of bytes per element
     * specified by the `type_info()` function.
     *
     * @return The total number of bytes in the buffer.
     */
    RPY_NO_DISCARD virtual dimn_t bytes() const;

    /**
     * @brief Migrates data to a specified device.
     *
     * This method migrates data from the current buffer to the specified
     * destination buffer on a specified device and queue. It returns an event
     * object that can be used to track the progress of the data migration.
     *
     * @param dst The destination buffer where the data will be migrated to.
     * @param device The device where the migration will take place.
     * @param queue The queue on which the migration will be performed.
     *
     * @return An event object that can be used to track the progress of the
     * data migration.
     *
     * @throws std::runtime_error if the data cannot be migrated to the
     * specified device.
     */
    RPY_NO_DISCARD virtual Event
    to_device(Buffer& dst, const Device& device, Queue& queue) const;

    /**
     * @brief Maps a mutable portion of the buffer.
     *
     * This method maps a mutable portion of the buffer starting from the
     * specified offset with the specified size into the CPU memory space.
     * It returns a new Buffer objec representing the mapped portion of the
     * buffer.
     *
     * @param size The size of the portion to map.
     * @param offset The offset to start mapping from.
     *
     * @return A new Buffer object representing the mapped portion of the
     * buffer.
     *
     * @see Buffer
     */
    RPY_NO_DISCARD virtual Buffer map_mut(dimn_t size, dimn_t offset);
    /**
     * @brief Maps a mutable portion of the buffer.
     *
     * This method maps a mutable portion of the buffer starting from the
     * specified offset with the specified size into the CPU memory space.
     * It returns a new Buffer object representing the mapped portion of the
     * buffer.
     *
     * @param size The size of the portion to map.
     * @param offset The offset to start mapping from.
     *
     * @return A new Buffer object representing the mapped portion of the
     * buffer.
     *
     * @see Buffer
     */
    RPY_NO_DISCARD virtual Buffer map(dimn_t size, dimn_t offset) const;

    /**
     * @brief Unmaps a mutable portion of the buffer.
     *
     * This method unmaps a mutable portion of the buffer represented by `ptr`.
     * It releases the CPU memory space previously mapped by the `map` method.
     * It does not throw any exceptions and is guaranteed to be noexcept.
     *
     * @param ptr The BufferInterface object representing the mapped portion of
     * the buffer to unmap.
     *
     * @see BufferInterface::map()
     */
    virtual void unmap(BufferInterface& ptr) const noexcept;

    /**
     * @brief Returns the memory owner of the BufferInterface object.
     *
     * The memory owner is a Buffer object that holds a counted reference to
     * the current BufferInterface object. This method is `const` and
     * `noexcept`.
     *
     * @return The memory owner of the BufferInterface object.
     */
    virtual Buffer memory_owner() const noexcept;

    /**
     * @brief Slices the buffer object.
     *
     * This method slices the buffer object by specifying the size and offset.
     *
     * @param size The size of the slice to be created.
     * @param offset The offset from the beginning of the buffer to start the
     * slice.
     * @return The sliced buffer object.
     */
    virtual Buffer slice(dimn_t size, dimn_t offset) const;
    /**
     * @brief Create a mutable sub-slice of the buffer.
     *
     * This method creates a mutable sub-slice of the buffer starting from the
     * given `offset` and with the specified `size`.
     *
     * @param size The size of the sub-slice to create.
     * @param offset The offset from which to start the sub-slice.
     *
     * @return A mutable sub-slice of the buffer.
     */
    virtual Buffer mut_slice(dimn_t size, dimn_t offset);
};

#ifdef RPY_PLATFORM_WINDOWS
#  ifdef RoughPy_Platform_EXPORTS
namespace dtl {
extern template class ObjectBase<BufferInterface, Buffer>;
}
#  else
namespace dtl {
template class RPY_DLL_IMPORT ObjectBase<BufferInterface, Buffer>;
}
#  endif
#else
namespace dtl {
extern template class ROUGHPY_DEVICES_EXPORT
        ObjectBase<BufferInterface, Buffer>;
}
#endif

namespace dtl {

template <typename T>
class BufferRange;

}

/**
 * @class Buffer
 * @brief Buffer class for manipulating buffer objects.
 *
 * The Buffer class provides methods for manipulating buffer objects. It is
 * derived from the ObjectBase class with the BufferInterface and Buffer as
 * template arguments.
 */
class ROUGHPY_DEVICES_EXPORT Buffer
    : public dtl::ObjectBase<BufferInterface, Buffer>
{
    using base_t = dtl::ObjectBase<BufferInterface, Buffer>;

public:
    using base_t::base_t;

    Buffer(Device device, dimn_t size, TypePtr type);
    Buffer(dimn_t size, TypePtr type);
    Buffer(Device device, void* ptr, dimn_t size, TypePtr type);
    Buffer(Device device, const void* ptr, dimn_t size, TypePtr type);

    Buffer(void* ptr, dimn_t size, TypePtr info);
    Buffer(const void* ptr, dimn_t size, TypePtr info);

    template <typename T>
    explicit Buffer(Device device, Slice<T> data);

    template <typename T>
    explicit Buffer(Device device, Slice<const T> data);

    template <typename T>
    explicit Buffer(Slice<T> data);

    template <typename T>
    explicit Buffer(Slice<const T> data);

    Buffer(TypePtr tp, dimn_t size);
    Buffer(TypePtr tp, void* ptr, dimn_t size);
    Buffer(TypePtr tp, const void* ptr, dimn_t size);
    Buffer(TypePtr tp, dimn_t size, Device device);
    Buffer(TypePtr tp, void* ptr, dimn_t size, Device device);
    Buffer(TypePtr tp, const void* ptr, dimn_t size, Device device);
    /**
     * @brief Returns the content type of the buffer.
     *
     * This method returns a pointer to the content type of the buffer. If the
     * buffer is null, it returns nullptr.
     *
     * @return const Type* - A pointer to the content type of the buffer.
     * @note The pointer should not be deleted and can become invalid if the
     * buffer is modified.
     */
    RPY_NO_DISCARD TypePtr type() const noexcept
    {
        return is_null() ? nullptr : impl()->type();
    }
    /**
     * @brief Get the size of the buffer.
     *
     * This method returns the size of the buffer. If the buffer implementation
     * is nullptr, it returns 0.
     *
     * @return dimn_t - The size of the buffer.
     */
    RPY_NO_DISCARD dimn_t size() const;
    /**
     * @brief Returns the size of the buffer in bytes.
     *
     * This method returns the size of the buffer in bytes. It calculates the
     * size by multiplying the size returned by the `size()` method of the
     * underlying implementation object with the number of bytes per element,
     * which is obtained from the `type_info().bytes` property of the
     * implementation object.
     *
     * @return The size of the buffer in bytes as a `dimn_t` value.
     */
    RPY_NO_DISCARD dimn_t bytes() const;
    /**
     * @brief Get the mode of the buffer.
     *
     * This method retrieves the current mode of the buffer object. If the
     * buffer object's implementation is null, it returns the default mode which
     * is `BufferMode::Read`.
     *
     * @return BufferMode - The mode of the buffer object. Possible values are
     * `BufferMode::Read` and `BufferMode::Write`.
     */
    RPY_NO_DISCARD BufferMode mode() const;
    /**
     * @brief Checks if the buffer is empty.
     *
     * This method checks if the buffer is empty by checking if it is
     * null or if the size of the buffer is zero.
     *
     * @return \c true if the buffer is empty, \c false otherwise.
     *
     * @note This method does not throw any exceptions.
     */
    RPY_NO_DISCARD bool empty() const noexcept
    {
        return is_null() || impl()->size() == 0;
    }

    /**
     * @brief Returns a slice of type `Slice<const T>`.
     *
     * This method returns a slice containing the elements of the buffer object.
     * The returned slice has type `Slice<const T>`, where `T` is the content
     * type of the buffer.
     *
     * @return Slice<const T> - A slice containing the elements of the buffer
     * object.
     *
     * @note The returned slice is read-only and cannot be modified through it.
     * The pointer to the slice's data should not be deleted or modified, as it
     * points to the internal data of the buffer object.
     *
     * @see `Slice`
     */
    template <typename T>
    Slice<const T> as_slice() const
    {
        return {static_cast<const T*>(ptr()), size()};
    }

    /**
     * @brief Get a mutable slice of the buffer.
     *
     * This method returns a `Slice<T>` object representing a mutable slice of
     * the buffer. It checks if the buffer mode is not set to `BufferMode::Read`
     * before returning the slice.
     *
     * @return A `Slice<T>` object representing a mutable slice of the buffer.
     */
    template <typename T>
    Slice<T> as_mut_slice()
    {
        RPY_CHECK(mode() != BufferMode::Read);
        return {static_cast<T*>(ptr()), size()};
    }

    /**
     * @brief Creates a new buffer object that represents a slice of the current
     * buffer, starting at the given offset and with the specified size.
     *
     * @param offset The offset in bytes at which the slice should start.
     * @param size The size in bytes of the slice.
     *
     * @return A new buffer object representing the slice. If the current buffer
     * is null or empty, an empty buffer object will be returned.
     *
     * @see Buffer
     */
    RPY_NO_DISCARD Buffer slice(dimn_t offset, dimn_t size);

    /**
     * @brief Slices the buffer object.
     *
     * This method slices the buffer object and returns a new buffer object with
     * the specified offset and size. The offset parameter determines the
     * starting position of the slice, and the size parameter determines the
     * size of the slice.
     *
     * @param offset The starting position of the slice.
     * @param size The size of the slice.
     *
     * @return A new buffer object that represents the slice of the original
     * buffer.
     */
    RPY_NO_DISCARD Buffer slice(dimn_t offset, dimn_t size) const;

    RPY_NO_DISCARD Slice<const Value> as_value_slice() const noexcept;

    RPY_NO_DISCARD Slice<Value> as_mut_value_slice();

    /**
     * @brief Transfers the contents of this buffer to the specified destination
     * buffer on the given device.
     *
     * This method transfers the contents of this buffer to the specified
     * destination buffer on the given device. The destination buffer must be
     * compatible with the source buffer in terms of size and format. The
     * transfer is performed asynchronously on the default queue of the device.
     * This method blocks until the transfer operation is complete.
     *
     * @param dst The destination buffer to which the contents of this buffer
     * will be transferred.
     * @param device The device on which the transfer will be performed.
     *
     * @sa Buffer, Device, check_device_compatibility
     */
    void to_device(Buffer& dst, const Device& device) const;

    /**
     * @brief Moves the content of the current buffer to the specified buffer,
     * on the specified device and queue.
     *
     * This function checks if the current buffer and destination buffer are
     * compatible with the specified device. If they are compatible, the content
     * of the current buffer is moved to the destination buffer on the specified
     * device and queue.
     *
     * @param dst The destination buffer to which the content will be moved.
     * @param device The device on which the destination buffer resides.
     * @param queue The queue on which the content will be moved.
     *
     * @return An instance of the `Event` class representing the event
     * associated with the buffer transfer operation. If the current buffer or
     * destination buffer is not valid, or if they are not compatible with the
     * specified device, an empty `Event` object is returned.
     */
    Event to_device(Buffer& dst, const Device& device, Queue& queue) const;

    /**
     * @brief Maps a portion of the buffer into memory.
     *
     * This method maps a portion of the buffer into memory, starting from the
     * given offset and with the specified size. If the size is set to 0, it
     * will map the entire buffer.
     *
     * @param size The size of the portion to map. If set to 0, it will map the
     * entire buffer.
     * @param offset The offset from which to start mapping.
     * @return The mapped portion of the buffer as a new Buffer object.
     *
     * @note The returned Buffer object will have its own reference to the
     * mapped memory. This means that the original buffer should not be modified
     * while the mapped portion is being used.
     */
    RPY_NO_DISCARD Buffer map(dimn_t size = 0, dimn_t offset = 0) const;
    /**
     * @brief Maps a portion of the buffer into memory.
     *
     * This method maps a portion of the buffer into memory and returns a new
     * buffer object representing the mapped region. If the implementation of
     * the buffer object is nullptr, it returns an empty buffer object. If the
     * size parameter is set to 0, it will use the size of the current buffer.
     *
     * @param size The size of the region to map. If set to 0, it will use the
     * size of the current buffer.
     * @param offset The offset from the beginning of the buffer to start
     * mapping from.
     *
     * @return A new buffer object representing the mapped region, or an empty
     * buffer object if the implementation is nullptr.
     */
    RPY_NO_DISCARD Buffer map(dimn_t size = 0, dimn_t offset = 0);

    /**
     * @brief Get the memory owner of the buffer.
     *
     * This method returns the memory owner of the buffer. If the buffer
     * implementation is nullptr, it returns a default-constructed Buffer
     * object.
     *
     * @return The memory owner of the buffer.
     */
    Buffer memory_owner() const noexcept;

    /**
     * @brief Check if the current object is the owner of the memory.
     *
     * This method compares the implementation of the memory owner with the
     * implementation of the current object. It returns true if they are the
     * same and false otherwise.
     *
     * @return True if the current object is the memory owner, false otherwise.
     *
     * @note This implementation is provided as an example and is not
     * recommended for production code. It is intended for demonstration and
     * testing purposes only.
     */
    bool is_owner() const noexcept
    {
        // This is a really bad implementation, but it will do for now
        return memory_owner().impl() == impl();
    }

    template <typename T>
    RPY_NO_DISCARD enable_if_t<
            is_same_v<T, Reference> || is_same_v<T, ConstReference>,
            dtl::BufferRange<T>>
    as_range() const;
};

template <typename T>
Buffer::Buffer(rpy::devices::Device device, Slice<T> data)
    : Buffer(device, data.data(), data.size(), devices::type_info<T>())
{}

template <typename T>
Buffer::Buffer(rpy::devices::Device device, Slice<const T> data)
    : Buffer(device, data.data(), data.size(), devices::type_info<T>())
{}

template <typename T>
Buffer::Buffer(Slice<T> data)
    : Buffer(get_host_device(),
             data.data(),
             data.size(),
             devices::type_info<T>())
{}

template <typename T>
Buffer::Buffer(Slice<const T> data)
    : Buffer(get_host_device(),
             data.data(),
             data.size(),
             devices::type_info<T>())
{}

namespace dtl {

template <typename T>
class BufferRangeIterator;

template <typename T>
class BufferRange
{
    Buffer m_buffer;

public:
    using value_type = T;
    using reference = T&;
    using const_reference = add_const_t<T>&;
    using pointer = T*;
    using const_pointer = add_const_t<T>*;
    using size_type = dimn_t;
    using difference_type = idimn_t;

    using iterator = BufferRangeIterator<T>;
    using const_iterator = BufferRangeIterator<const T>;

    iterator begin() noexcept;
    iterator end() noexcept;
    const_iterator begin() const noexcept;
    const_iterator cbegin() const noexcept;
    const_iterator end() const noexcept;
    const_iterator cend() const noexcept;
};

template <>
class BufferRange<Reference>
{
    Buffer m_buffer;

public:
    using value_type = Reference;
    using reference = Reference;
    using const_reference = Reference;
    using pointer = void*;
    using const_pointer = const void*;
    using size_type = dimn_t;
    using difference_type = idimn_t;

    using iterator = BufferRangeIterator<Reference>;
    using const_iterator = BufferRangeIterator<ConstReference>;
};

template <typename T>
class BufferRangeIterator
{
    T* m_ptr;

public:
    using value_type = T;
    using reference = T&;
    using pointer = T*;
    using size_type = dimn_t;
    using difference_type = idimn_t;

    using iterator_category = std::random_access_iterator_tag;

    explicit constexpr BufferRangeIterator(T* ptr) : m_ptr(ptr) {}

    BufferRangeIterator& operator++() noexcept
    {
        ++m_ptr;
        return *this;
    }

    const BufferRangeIterator operator++(int) noexcept
    {
        BufferRangeIterator prev(*this);
        ++m_ptr;
        return prev;
    }

    BufferRangeIterator& operator--() noexcept
    {
        --m_ptr;
        return *this;
    }

    const BufferRangeIterator operator--(int) noexcept
    {
        BufferRangeIterator prev(*this);
        --m_ptr;
        return prev;
    }

    T& operator*() const noexcept { return *m_ptr; }
    T* operator->() const noexcept { return m_ptr; }

    constexpr BufferRangeIterator& operator+=(difference_type n) noexcept
    {
        m_ptr += n;
        return *this;
    }

    constexpr BufferRangeIterator& operator-=(difference_type n) noexcept
    {
        m_ptr -= n;
        return *this;
    }

    constexpr BufferRangeIterator operator+(difference_type n) const noexcept
    {
        return BufferRangeIterator(m_ptr + 1);
    }

    constexpr BufferRangeIterator operator-(difference_type n) const noexcept
    {
        return BufferRangeIterator(m_ptr - 1);
    }

    constexpr reference operator[](difference_type n) noexcept
    {
        return m_ptr[n];
    }

    friend constexpr bool operator!=(
            const BufferRangeIterator& left,
            const BufferRangeIterator& right
    ) noexcept
    {
        return left.m_ptr == right.m_ptr;
    }

    friend constexpr difference_type operator-(
            const BufferRangeIterator& left,
            const BufferRangeIterator& right
    ) noexcept
    {
        return static_cast<difference_type>(right - left);
    }

    friend constexpr bool operator<(
            const BufferRangeIterator& left,
            const BufferRangeIterator& right
    ) noexcept
    {
        return left.m_ptr < right.m_ptr;
    }

    friend constexpr bool operator<=(
            const BufferRangeIterator& left,
            const BufferRangeIterator& right
    ) noexcept
    {
        return left.m_ptr <= right.m_ptr;
    }

    friend constexpr bool operator>(
            const BufferRangeIterator& left,
            const BufferRangeIterator& right
    ) noexcept
    {
        return left.m_ptr > right.m_ptr;
    }

    friend constexpr bool operator>=(
            const BufferRangeIterator& left,
            const BufferRangeIterator& right
    ) noexcept
    {
        return left.m_ptr >= right.m_ptr;
    }
};

template <typename T>
constexpr bool operator==(
        const BufferRangeIterator<T>& left,
        const BufferRangeIterator<T>& right
) noexcept
{
    return !(left != right);
}

template <typename I, typename T>
constexpr BufferRangeIterator<T>
operator+(I left, const BufferRangeIterator<T>& right) noexcept
{
    using difference_type = typename BufferRangeIterator<T>::difference_type;
    return right + static_cast<difference_type>(left);
}

template <typename T>
typename BufferRange<T>::iterator BufferRange<T>::begin() noexcept
{
    return iterator(static_cast<T*>(m_buffer.ptr()));
}
template <typename T>
typename BufferRange<T>::iterator BufferRange<T>::end() noexcept
{
    return iterator(static_cast<T*>(m_buffer.ptr())) + m_buffer.size();
}
template <typename T>
typename BufferRange<T>::const_iterator BufferRange<T>::begin() const noexcept
{
    return iterator(static_cast<T*>(m_buffer.ptr()));
}
template <typename T>
typename BufferRange<T>::const_iterator BufferRange<T>::cbegin() const noexcept
{
    return iterator(static_cast<T*>(m_buffer.ptr()));
}
template <typename T>
typename BufferRange<T>::const_iterator BufferRange<T>::end() const noexcept
{
    return iterator(static_cast<T*>(m_buffer.ptr())) + m_buffer.size();
}
template <typename T>
typename BufferRange<T>::const_iterator BufferRange<T>::cend() const noexcept
{
    return iterator(static_cast<T*>(m_buffer.ptr())) + m_buffer.size();
}

}// namespace dtl

}// namespace devices

namespace dtl {

template <typename T>
class SliceIterator
{
    devices::TypePtr p_type;
    copy_cv_t<T, byte>* p_data;
};

}// namespace dtl

template <>
class Slice<const devices::Value>
{
    devices::Buffer m_buffer;
    const void* p_data;
    dimn_t m_size;

public:
    explicit Slice(devices::Buffer buffer)
        : m_buffer(std::move(buffer)),
          p_data(m_buffer.ptr()),
          m_size(m_buffer.size())
    {}
};

template <>
class Slice<devices::Value>
{
    devices::Buffer m_buffer;
    void* p_data;
    dimn_t m_size;

public:
    explicit Slice(devices::Buffer buffer)
        : m_buffer(std::move(buffer)),
          p_data(buffer.ptr()),
          m_size(buffer.size())
    {}
};

namespace devices {

inline Slice<const Value> Buffer::as_value_slice() const noexcept
{
    return Slice<const Value>(this->map());
}
inline Slice<Value> Buffer::as_mut_value_slice()
{
    return Slice<Value>(this->map());
}

}// namespace devices

}// namespace rpy
#endif// ROUGHPY_DEVICE_BUFFER_H_
