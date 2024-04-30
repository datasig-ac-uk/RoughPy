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

#ifndef ROUGHPY_SCALARS_SCALAR_ARRAY_H_
#define ROUGHPY_SCALARS_SCALAR_ARRAY_H_

#include "packed_scalar_type_ptr.h"
#include "scalar_type.h"
#include "scalars_fwd.h"
#include "traits.h"

#include <roughpy/core/container/vector.h>
#include <roughpy/devices/buffer.h>
#include <roughpy/platform/serialization.h>

namespace rpy {
namespace scalars {

namespace dtl {
enum class ScalarArrayStorageModel
{
    BorrowConst = 0,
    BorrowMut = 1,
    Owned = 2,
};

}

/**
 * @class ScalarArray
 * @brief Represents an array of scalar values.
 *
 * The ScalarArray class represents an array of scalar values. It provides
 * various methods to manipulate and access the scalar array. The scalar array
 * can be created with different types and sizes, and it can be copied, moved,
 * assigned, and accessed using slice indexes. The ScalarArray class also
 * provides methods to retrieve the type, size, capacity, and ownership status
 * of the scalar array.
 */
class ROUGHPY_SCALARS_EXPORT ScalarArray
{

    // Scalar type is only necessary if the buffer type is not fundamental,
    // in which case it must not be null
    const ScalarType* p_type = nullptr;

    // All memory is represented as a buffer now, even if it is a borrowed
    // pointer from some other place. This should dramatically simplify things.
    devices::Buffer m_buffer;

public:
    ScalarArray();
    ScalarArray(const ScalarArray& other);
    ScalarArray(ScalarArray&& other) noexcept;

    explicit ScalarArray(const ScalarType* type, dimn_t size = 0);
    explicit ScalarArray(devices::TypeInfo info, dimn_t size = 0);

    ScalarArray(PackedScalarType type, const void* data, dimn_t size);
    ScalarArray(PackedScalarType type, void* data, dimn_t size);
    ScalarArray(PackedScalarType type, devices::Buffer&& buffer);

    template <typename T>
    explicit ScalarArray(Slice<T> data);

    template <typename T>
    explicit ScalarArray(Slice<const T> data);

    template <typename T>
    ScalarArray(T* data, dimn_t size);

    template <typename T>
    ScalarArray(const T* data, dimn_t size);

    ~ScalarArray();

    ScalarArray& operator=(const ScalarArray& other);
    ScalarArray& operator=(ScalarArray&& other) noexcept;

    ScalarArray copy_or_clone() &&;

    /**
     * @brief Checks if the ScalarArray is owning its buffer.
     *
     * This method returns true if the ScalarArray is owning its buffer,
     * and false otherwise. The ownership status is determined by the underlying
     * buffer of the ScalarArray.
     *
     * @return True if the ScalarArray is owning its buffer, false otherwise.
     */
    RPY_NO_DISCARD bool is_owning() const noexcept
    {
        return m_buffer.is_owner();
    }

    /**
     * @brief Retrieves the type of the packed scalar values in the ScalarArray.
     *
     * This method returns the PackedScalarType of the packed scalar values
     * in the ScalarArray. The type represents the data type of the scalar
     * values stored in the array.
     *
     * @return The PackedScalarType of the packed scalar values.
     * @note The return value is only valid if the ScalarArray has been
     * initialized with valid packed scalar values. If the ScalarArray is not
     * initialized, the return value may be invalid.
     * @remark The returned PackedScalarType value points to either a
     * pre-defined type or a custom-defined type. It is recommended to use the
     * appropriate access methods of the PackedScalarType class to retrieve the
     * specific details and properties of the scalar type, such as size,
     * precision, signed/unsigned, etc. Please refer to the PackedScalarType
     * class documentation for more information.
     * @remark The return type is constant and noexcept guaranteeing that the
     * method will not throw any exceptions. However, the behavior is undefined
     * if the ScalarArray is not valid.
     */
    RPY_NO_DISCARD PackedScalarType type() const noexcept;

    /**
     * @brief Returns the type information of the ScalarArray.
     *
     * This method returns the type information of the ScalarArray.
     * It accesses the type information of the internal buffer and returns it.
     *
     * @return The type information of the ScalarArray.
     */
    RPY_NO_DISCARD devices::TypeInfo type_info() const noexcept;

    /**
     * @brief Retrieves the size of the ScalarArray.
     *
     * This method returns the size of the ScalarArray, which represents the
     * number of elements in the array. The size is determined by the size of
     * the internal buffer of the ScalarArray.
     *
     * @return The size of the ScalarArray.
     * @note The return value represents the number of elements in the array and
     * is not related to the memory occupied by the array.
     * @remark The return type is constant and noexcept guaranteeing that the
     * method will not throw any exceptions.
     */
    RPY_NO_DISCARD dimn_t size() const noexcept { return m_buffer.size(); }
    RPY_NO_DISCARD dimn_t capacity() const noexcept;
    /**
     * @fn bool empty() const noexcept
     * @brief Checks if the scalar array is empty.
     *
     * This function checks whether the scalar array is empty or not. An empty
     * scalar array has no elements.
     *
     * @return True if the scalar array is empty, false otherwise.
     *
     * @note This function does not modify the state of the object.
     * @note This function should be called on a valid scalar array object.
     * @note This function has a constant time complexity.
     * @note This function is safe to use in a noexcept context.
     */
    RPY_NO_DISCARD bool empty() const noexcept { return m_buffer.size() == 0; }
    /**
     * @brief Checks if the scalar array is null.
     *
     * This method checks if the scalar array is null by calling the `is_null()`
     * method of the internal buffer.
     *
     * @return True if the scalar array is null, false otherwise.
     *
     * @see scalar_array.h
     */
    RPY_NO_DISCARD bool is_null() const noexcept { return m_buffer.is_null(); }
    /**
     * @brief Checks if the ScalarArray is read-only.
     *
     * This method checks if the ScalarArray is read-only by comparing the mode
     * of the underlying buffer with devices::BufferMode::Read. If the mode is
     * devices::BufferMode::Read, then the ScalarArray is considered to be
     * read-only. Otherwise, it is not read-only.
     *
     * @return True if the ScalarArray is read-only, false otherwise.
     */
    RPY_NO_DISCARD bool is_const() const noexcept
    {
        return m_buffer.mode() == devices::BufferMode::Read;
    }
    /**
     * @brief Retrieves the device associated with the device.
     *
     * This method returns the device associated with the device.
     *
     * @return The device associated with the device.
     *
     * @note This method does not throw any exceptions.
     */
    RPY_NO_DISCARD devices::Device device() const noexcept
    {
        return m_buffer.device();
    }

    /**
     * @brief Returns the memory owner of the buffer.
     *
     * This method returns the memory owner of the buffer associated with the
     * current object. The returned memory owner is of type devices::Buffer.
     *
     * @return The memory owner of the buffer.
     */
    RPY_NO_DISCARD devices::Buffer memory_owner() const noexcept
    {
        return m_buffer.memory_owner();
    }

    /**
     * @brief Returns a reference to the buffer object of the ScalarArray.
     *
     * This method returns a constant reference to the buffer object of the
     * ScalarArray. The buffer object is used to store the scalar values of the
     * array.
     *
     * @return A constant reference to the buffer object.
     */
    RPY_NO_DISCARD const devices::Buffer& buffer() const;
    /**
     * @fn devices::Buffer& ScalarArray::mut_buffer()
     * @brief Retrieves a mutable reference to the buffer object associated with
     * the ScalarArray.
     *
     * This function returns a mutable reference to the buffer object associated
     * with the ScalarArray. The buffer object represents the underlying data
     * storage for the ScalarArray. By retrieving a mutable reference to the
     * buffer, you can directly manipulate the data stored in the ScalarArray.
     *
     * @return A mutable reference to the buffer object associated with the
     * ScalarArray.
     */
    RPY_NO_DISCARD devices::Buffer& mut_buffer();

    RPY_NO_DISCARD Scalar operator[](dimn_t i) const;
    RPY_NO_DISCARD Scalar operator[](dimn_t i);

    RPY_NO_DISCARD ScalarArray operator[](SliceIndex index);
    RPY_NO_DISCARD ScalarArray operator[](SliceIndex index) const;

    /**
     * @brief Returns a view of the ScalarArray.
     *
     * This method returns a view of the ScalarArray. The view is created by
     * mapping the buffer of the ScalarArray to the desired size.
     *
     * @return A view of the ScalarArray.
     */
    RPY_NO_DISCARD ScalarArray view() const
    {
        return {p_type, m_buffer.map(size())};
    }

    /**
     * @brief Provides a mutable view of the ScalarArray.
     *
     * This method returns a mutable view of the ScalarArray by creating a new
     * ScalarArray with the same type and a mapped buffer of the same size. The
     * view allows modifying the contents of the original ScalarArray without
     * creating a deep copy.
     *
     * @return A mutable view of the ScalarArray.
     */
    RPY_NO_DISCARD ScalarArray mut_view()
    {
        return {p_type, m_buffer.map(size())};
    }

    /**
     * @brief Returns a borrowed copy of the ScalarArray object.
     *
     * This method returns a borrowed copy of the current ScalarArray object.
     * The borrowed copy shares the same underlying data buffer with the
     * original object. Changes made to the borrowed copy will affect the
     * original object and vice versa.
     *
     * @return A borrowed copy of the ScalarArray object.
     */
    RPY_NO_DISCARD ScalarArray borrow() const;
    /**
     * @fn ScalarArray::borrow_mut()
     * @brief Borrows a mutable reference to the scalar array.
     *
     * This method returns a mutable reference to the scalar array. It allows
     * for modification of the array elements.
     *
     * @return A mutable reference to the scalar array.
     */
    RPY_NO_DISCARD ScalarArray borrow_mut();

    /**
     * @brief Converts the ScalarArray to the specified device.
     *
     * This method creates a new ScalarArray object that is stored on the
     * specified device. If the existing ScalarArray is already stored on the
     * specified device, it returns a copy of the existing ScalarArray object.
     *
     * @param device The device to which the ScalarArray will be moved.
     * @return A ScalarArray object that is stored on the specified device.
     */
    RPY_NO_DISCARD ScalarArray to_device(devices::Device device) const;

    RPY_SERIAL_SAVE_FN();
    RPY_SERIAL_LOAD_FN();

private:
    void check_for_ptr_access(bool mut = false) const;
    RPY_NO_DISCARD containers::Vec<byte> to_raw_bytes() const;
    void from_raw_bytes(PackedScalarType type, dimn_t count, Slice<byte> bytes);

public:
    //    template <typename T>
    //    RPY_NO_DISCARD Slice<T> as_mut_slice()
    //    {
    //        check_for_ptr_access(true);
    //        return {static_cast<T*>(raw_mut_pointer()), m_size};
    //    }
    //
    //    template <typename T>
    //    RPY_NO_DISCARD Slice<const T> as_slice() const
    //    {
    //        check_for_ptr_access(false);
    //        return {static_cast<const T*>(raw_pointer()), m_size};
    //    }

private:
    //    RPY_NO_DISCARD const void* raw_pointer(dimn_t i = 0) const noexcept;
    //    RPY_NO_DISCARD void* raw_mut_pointer(dimn_t i = 0) noexcept;
};

template <typename T>
ScalarArray::ScalarArray(Slice<T> data) : m_buffer(data)
{
    const auto info = m_buffer.type_info();
    if (!traits::is_fundamental(info)) {
        auto tp = scalar_type_of<T>();
        RPY_CHECK(tp);
        p_type = *tp;
    }
}

template <typename T>
ScalarArray::ScalarArray(Slice<const T> data) : m_buffer(data)
{
    const auto info = m_buffer.type_info();
    if (!traits::is_fundamental(info)) {
        auto tp = scalar_type_of<T>();
        RPY_CHECK(tp);
        p_type = *tp;
    }
}

template <typename T>
ScalarArray::ScalarArray(T* data, dimn_t size) : m_buffer({data, size})
{
    const auto info = m_buffer.type_info();
    if (!traits::is_fundamental(info)) {
        auto tp = scalar_type_of<T>();
        RPY_CHECK(tp);
        p_type = *tp;
    }
}
template <typename T>
ScalarArray::ScalarArray(const T* data, dimn_t size) : m_buffer({data, size})
{
    const auto info = m_buffer.type_info();
    if (!traits::is_fundamental(info)) {
        auto tp = scalar_type_of<T>();
        RPY_CHECK(tp);
        p_type = *tp;
    }
}

RPY_SERIAL_LOAD_FN_IMPL(ScalarArray)
{
    devices::TypeInfo type_info;
    // RPY_SERIAL_SERIALIZE_VAL(type_info);
    uint64_t count;
    RPY_SERIAL_SERIALIZE_VAL(count);
    containers::Vec<byte> raw_bytes;
    RPY_SERIAL_SERIALIZE_VAL(raw_bytes);
    // from_raw_bytes(type_info, count, raw_bytes);
}
RPY_SERIAL_SAVE_FN_IMPL(ScalarArray)
{
    // RPY_SERIAL_SERIALIZE_NVP("type_info", type_info());
    RPY_SERIAL_SERIALIZE_NVP("count", static_cast<uint64_t>(size()));
    RPY_SERIAL_SERIALIZE_NVP("raw_bytes", to_raw_bytes());
}

#ifdef RPY_COMPILING_SCALARS
RPY_SERIAL_EXTERN_SAVE_CLS_BUILD(ScalarArray)
RPY_SERIAL_EXTERN_LOAD_CLS_BUILD(ScalarArray)
#else
RPY_SERIAL_EXTERN_SAVE_CLS_IMP(ScalarArray)
RPY_SERIAL_EXTERN_LOAD_CLS_IMP(ScalarArray)
#endif

/**
 * @brief Converts and copies the contents of one ScalarArray to another.
 *
 * This function copies the scalar values from the source ScalarArray to the
 * destination ScalarArray, converting them if necessary. The destination
 * ScalarArray must be pre-allocated with enough capacity to hold all the
 * elements of the source ScalarArray.
 *
 * @param dst The destination ScalarArray to copy the elements to.
 * @param src The source ScalarArray to copy the elements from.
 *
 * @note The size and capacity of the destination ScalarArray will be modified
 * to match that of the source ScalarArray.
 * @note The data in the destination ScalarArray will be overwritten.
 * @note Both the source and destination ScalarArrays must have the same length.
 */
ROUGHPY_SCALARS_EXPORT void
convert_copy(ScalarArray& dst, const ScalarArray& src);

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SCALAR_ARRAY_H_
