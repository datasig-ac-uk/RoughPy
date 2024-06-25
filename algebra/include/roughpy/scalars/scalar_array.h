//
// Created by sam on 24/06/24.
//

#ifndef ROUGHPY_SCALARS_SCALAR_ARRAY_H
#define ROUGHPY_SCALARS_SCALAR_ARRAY_H

#include "scalars_fwd.h"

#include <roughpy/devices/buffer.h>
#include <roughpy/devices/core.h>
#include <roughpy/devices/device_handle.h>
#include <roughpy/devices/host_device.h>
#include <roughpy/devices/type.h>

namespace rpy {
namespace scalars {

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
class ROUGHPY_SCALARS_EXPORT ScalarArray : public devices::Buffer
{
public:
    using Buffer::Buffer;

    explicit ScalarArray(TypePtr type, dimn_t size = 0)
        : Buffer(std::move(type), size, devices::get_host_device())
    {}

    ScalarArray(TypePtr type, const void* data, dimn_t size)
        : Buffer(std::move(type), data, size)
    {}
    ScalarArray(TypePtr type, void* data, dimn_t size)
        : Buffer(std::move(type), data, size)
    {}

    explicit ScalarArray(Buffer&& buffer) noexcept : Buffer(std::move(buffer))
    {}

    template <typename T>
    explicit ScalarArray(Slice<T> data);

    template <typename T>
    explicit ScalarArray(Slice<const T> data);

    template <typename T>
    ScalarArray(T* data, dimn_t size);

    template <typename T>
    ScalarArray(const T* data, dimn_t size);

    /**
     * @brief Checks if the scalar array is empty.
     *
     * This function checks whether the scalar array is empty or not. An empty
     * scalar array has no elements.
     * @return True if the scalar array is empty, false otherwise.
     *
     * @note This function does not modify the state of the object.
     * @note This function should be called on a valid scalar array object.
     * @note This function has a constant time complexity.
     * @note This function is safe to use in a noexcept context.
     */
    RPY_NO_DISCARD bool empty() const noexcept { return size() == 0; }
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
        return mode() == devices::BufferMode::Read;
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

    RPY_NO_DISCARD ScalarCRef operator[](dimn_t i) const;
    RPY_NO_DISCARD ScalarRef operator[](dimn_t i);

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
    RPY_NO_DISCARD ScalarArray view() const { return ScalarArray(map(size())); }

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
    RPY_NO_DISCARD ScalarArray mut_view() { return ScalarArray(map(size())); }

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
    void from_raw_bytes(TypePtr type, dimn_t count, Slice<byte> bytes);

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
ScalarArray::ScalarArray(Slice<T> data) : Buffer(data)
{}

template <typename T>
ScalarArray::ScalarArray(Slice<const T> data) : Buffer(data)
{}

template <typename T>
ScalarArray::ScalarArray(T* data, dimn_t size) : Buffer({data, size})
{}
template <typename T>
ScalarArray::ScalarArray(const T* data, dimn_t size) : Buffer({data, size})
{}

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

#endif// ROUGHPY_SCALARS_SCALAR_ARRAY_H
