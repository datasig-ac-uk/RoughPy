//
// Created by sam on 3/31/24.
//

#ifndef ROUGHPY_DEVICES_ALGORITHMS_H
#define ROUGHPY_DEVICES_ALGORITHMS_H

#include "buffer.h"
#include "core.h"
#include "device_handle.h"
#include "type.h"
#include "value.h"

#include <roughpy/core/alloc.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include <roughpy/devices/type_dispatcher.h>

namespace rpy {
namespace devices {

/**
 * @class AlgorithmDrivers
 * @brief A class that provides various algorithms for buffer operations.
 *
 * This class implements several algorithms for performing operations on
 * buffers, such as finding values, counting occurrences, comparing buffers,
 * copying contents, etc. These algorithms are used to manipulate the contents
 * of the buffer and perform common operations efficiently.
 *
 * @sa Buffer
 */
class ROUGHPY_DEVICES_EXPORT AlgorithmDrivers
{
public:
    virtual ~AlgorithmDrivers();

    /**
     * @brief Find the first position where a value occurs in a buffer.
     *
     * This method is used to find the position of the first occurrence
     * of a specified value in a buffer. It returns an optional that contains
     * the position if the value is found, or an optional without a value if the
     * value is not found.
     *
     * @param buffer The buffer in which to search for the value.
     * @param value The value to search for.
     *
     * @return An optional containing the position of the first occurrence of
     * the specified value in the buffer. If the value is not found, an optional
     * without a value is returned.
     *
     * @sa Buffer
     */
    RPY_NO_DISCARD virtual optional<dimn_t>
    find(const Buffer& buffer, ConstReference value) const;

    /**
     * @brief Counts the occurrences of a specific value in the given buffer.
     *
     * This method takes a buffer and a value as input and returns the number of
     * occurrences of that value in the buffer. The buffer is expected to be of
     * type Buffer and the value is of type Reference. The returned value
     * indicates the total number of occurrences found in the buffer.
     *
     * @param buffer The buffer to search for occurrences of the value.
     * @param value The value to count in the buffer.
     * @return The number of occurrences of the value in the buffer.
     */
    RPY_NO_DISCARD virtual dimn_t
    count(const Buffer& buffer, ConstReference value) const;

    /**
     * @brief Find the first position where two buffers differ.
     *
     * This method compares the elements of the left and right buffers and
     * returns the position of the first mismatch. If the buffers are equal, it
     * returns an optional that does not contain a value.
     *
     * @param left The first buffer to compare.
     * @param right The second buffer to compare.
     *
     * @return An optional containing the position of the first mismatch if the
     * buffers are different. If the buffers are equal, an optional without a
     * value is returned.
     *
     * @sa Buffer
     */
    RPY_NO_DISCARD virtual optional<dimn_t>
    mismatch(const Buffer& left, const Buffer& right) const;

    /**
     * @brief Copy the contents of one buffer to another.
     *
     * This method copies the contents of the source buffer to the destination
     * buffer. The size and capacity of the destination buffer will be adjusted
     * to match the source buffer.
     *
     * @param dest The destination buffer to copy to.
     * @param source The source buffer from which to copy.
     *
     * @note The source buffer is not modified.
     *
     * @warning The destination buffer must have enough capacity to hold the
     * entire contents of the source buffer. Otherwise, undefined behavior may
     * occur.
     *
     * @sa Buffer
     */
    virtual void copy(Buffer& dest, const Buffer& source) const;

    /**
     * @brief Swaps the ranges of two buffers.
     *
     * This method swaps the ranges of two buffers, namely 'left' and 'right'.
     *
     * @param left The 'Buffer' object representing the left buffer.
     * @param right The 'Buffer' object representing the right buffer.
     *
     * @sa Buffer
     */
    virtual void swap_ranges(Buffer& left, Buffer& right) const;

    /**
     * @fn virtual void AlgorithmDrivers::fill(Buffer& dst, Reference value)
     * const
     * @brief Fills the destination buffer with the specified value.
     *
     * This method fills the destination buffer with the specified value.
     *
     * @param dst The destination buffer to be filled.
     * @param value The value to fill the buffer with.
     *
     * @sa Buffer
     */
    virtual void fill(Buffer& dst, ConstReference value) const;

    /**
     * @brief Reverses the contents of the given buffer.
     *
     * This method reverses the contents of the buffer in-place. The contents
     * of the buffer are modified and the reversed buffer is stored back
     * in the original buffer.
     *
     * @param buffer The buffer to be reversed.
     */
    virtual void reverse(Buffer& buffer) const;

    /**
     * @brief Shifts the contents of the provided buffer to the left.
     *
     * This method shifts the elements in the buffer to the left by one
     * position. The element at the first position is moved to the last
     * position.
     *
     * @param buffer The buffer to be shifted.
     *
     * @sa Buffer
     */
    virtual void shift_left(Buffer& buffer) const;

    /**
     * @brief Shifts the elements in the given buffer to the right.
     *
     * This method shifts the elements in the provided buffer to the right by
     * one position. The rightmost element will wrap around to the beginning of
     * the buffer.
     *
     * @param buffer The buffer to shift the elements in.
     *
     * @sa Buffer
     */
    virtual void shift_right(Buffer& buffer) const;

    /**
     * @fn AlgorithmDrivers::lower_bound(const Buffer& buffer, Reference value)
     * const
     * @brief Returns an optional index representing the lower bound of a value
     * in the buffer.
     *
     * This method uses the lower bound algorithm to find the index of the first
     * element in the buffer that is not less than the specified value. The
     * buffer is assumed to be sorted in ascending order. If the value is found
     * in the buffer, the index of the value is returned. Otherwise, the index
     * of the first element in the buffer greater than the value is returned. If
     * the buffer is empty or all elements are less than the specified value, an
     * empty optional is returned.
     *
     * @param buffer The buffer to search in. Must be sorted in ascending order.
     * @param value The value to search for.
     * @return An optional index representing the lower bound of the value. If
     * the value is found, the index of the value is returned. Otherwise, the
     * index of the first element greater than the value is returned. If the
     * buffer is empty or all elements are less than the value, an empty
     * optional is returned.
     *
     * @sa Buffer
     */
    virtual optional<dimn_t>
    lower_bound(const Buffer& buffer, ConstReference value) const;

    /**
     * @brief Find the upper bound of a value in a buffer.
     *
     * This method finds the position of the first element in the buffer that is
     * greater than the given value. The buffer is expected to be sorted in
     * non-decreasing order.
     *
     * @param buffer The buffer in which to search for the upper bound.
     * @param value The value for which to find the upper bound.
     * @return An optional containing the position of the upper bound if found,
     * or an empty optional if not found.
     * @sa Buffer
     */
    virtual optional<dimn_t>
    upper_bound(const Buffer& buffer, ConstReference value) const;

    /**
     * @brief Finds the maximum value in the given buffer and stores it in the
     * provided output parameter.
     *
     * This method performs an algorithm to find the maximum value in the given
     * buffer. It uses the `buffer` parameter as input and calculates the
     * maximum value using an internally implemented algorithm. The maximum
     * value is then stored in the `out` parameter.
     *
     * Note: The `out` parameter must be a modifiable reference.
     *
     * @param buffer The input buffer to search for the maximum value.
     * @param out    The output parameter to store the maximum value.
     *
     * @sa Buffer
     */
    virtual void max(const Buffer& buffer, Reference out) const;

    /**
     * @brief Find the minimum value in the given buffer and store the result in
     * the specified reference object.
     *
     * This method calculates the minimum value in the specified buffer by using
     * the MinFunctor algorithm. The result is then stored in the provided
     * reference object.
     *
     * @param buffer The buffer to find the minimum value from.
     * @param out The reference object to store the minimum value.
     *
     * @sa Buffer
     */
    virtual void min(const Buffer& buffer, Reference out) const;

    /**
     * @brief Compare two buffers lexicographically.
     *
     * The comparison is performed by
     * iterating over the corresponding elements in both buffers and comparing
     * them in lexicographical order. The elements are compared using their
     * underlying value type's comparison operator.
     *
     * @param left The left buffer to compare.
     * @param right The right buffer to compare.
     * @return True if the left buffer is lexicographically less than the right
     * buffer, false otherwise.
     *
     * @sa Buffer
     */
    virtual bool
    lexicographical_compare(const Buffer& left, const Buffer& right) const;
};

class AlgorithmsDispatcher : public TypeDispatcher<AlgorithmDrivers, void, void>
{
public:
    /**
     * @brief Find the first position where a value occurs in a buffer.
     *
     * This method is used to find the position of the first occurrence
     * of a specified value in a buffer. It returns an optional that contains
     * the position if the value is found, or an optional without a value if the
     * value is not found.
     *
     * @param buffer The buffer in which to search for the value.
     * @param value The value to search for.
     *
     * @return An optional containing the position of the first occurrence of
     * the specified value in the buffer. If the value is not found, an optional
     * without a value is returned.
     *
     * @sa Buffer
     */
    RPY_NO_DISCARD optional<dimn_t>
    find(const Buffer& buffer, ConstReference value) const;

    /**
     * @brief Counts the occurrences of a specific value in the given buffer.
     *
     * This method takes a buffer and a value as input and returns the number of
     * occurrences of that value in the buffer. The buffer is expected to be of
     * type Buffer and the value is of type Reference. The returned value
     * indicates the total number of occurrences found in the buffer.
     *
     * @param buffer The buffer to search for occurrences of the value.
     * @param value The value to count in the buffer.
     * @return The number of occurrences of the value in the buffer.
     */
    RPY_NO_DISCARD dimn_t
    count(const Buffer& buffer, ConstReference value) const;

    /**
     * @brief Find the first position where two buffers differ.
     *
     * This method compares the elements of the left and right buffers and
     * returns the position of the first mismatch. If the buffers are equal, it
     * returns an optional that does not contain a value.
     *
     * @param left The first buffer to compare.
     * @param right The second buffer to compare.
     *
     * @return An optional containing the position of the first mismatch if the
     * buffers are different. If the buffers are equal, an optional without a
     * value is returned.
     *
     * @sa Buffer
     */
    RPY_NO_DISCARD optional<dimn_t>
    mismatch(const Buffer& left, const Buffer& right) const;

    /**
     * @brief Copy the contents of one buffer to another.
     *
     * This method copies the contents of the source buffer to the destination
     * buffer. The size and capacity of the destination buffer will be adjusted
     * to match the source buffer.
     *
     * @param dest The destination buffer to copy to.
     * @param source The source buffer from which to copy.
     *
     * @note The source buffer is not modified.
     *
     * @warning The destination buffer must have enough capacity to hold the
     * entire contents of the source buffer. Otherwise, undefined behavior may
     * occur.
     *
     * @sa Buffer
     */
    void copy(Buffer& dest, const Buffer& source) const;

    /**
     * @brief Swaps the ranges of two buffers.
     *
     * This method swaps the ranges of two buffers, namely 'left' and 'right'.
     *
     * @param left The 'Buffer' object representing the left buffer.
     * @param right The 'Buffer' object representing the right buffer.
     *
     * @sa Buffer
     */
    void swap_ranges(Buffer& left, Buffer& right) const;

    /**
     * @fn virtual void AlgorithmDrivers::fill(Buffer& dst, Reference value)
     * const
     * @brief Fills the destination buffer with the specified value.
     *
     * This method fills the destination buffer with the specified value.
     *
     * @param dst The destination buffer to be filled.
     * @param value The value to fill the buffer with.
     *
     * @sa Buffer
     */
    void fill(Buffer& dst, ConstReference value) const;

    /**
     * @brief Reverses the contents of the given buffer.
     *
     * This method reverses the contents of the buffer in-place. The contents
     * of the buffer are modified and the reversed buffer is stored back
     * in the original buffer.
     *
     * @param buffer The buffer to be reversed.
     */
    void reverse(Buffer& buffer) const;

    /**
     * @brief Shifts the contents of the provided buffer to the left.
     *
     * This method shifts the elements in the buffer to the left by one
     * position. The element at the first position is moved to the last
     * position.
     *
     * @param buffer The buffer to be shifted.
     *
     * @sa Buffer
     */
    void shift_left(Buffer& buffer) const;

    /**
     * @brief Shifts the elements in the given buffer to the right.
     *
     * This method shifts the elements in the provided buffer to the right by
     * one position. The rightmost element will wrap around to the beginning of
     * the buffer.
     *
     * @param buffer The buffer to shift the elements in.
     *
     * @sa Buffer
     */
    void shift_right(Buffer& buffer) const;

    /**
     * @fn AlgorithmDrivers::lower_bound(const Buffer& buffer, Reference value)
     * const
     * @brief Returns an optional index representing the lower bound of a value
     * in the buffer.
     *
     * This method uses the lower bound algorithm to find the index of the first
     * element in the buffer that is not less than the specified value. The
     * buffer is assumed to be sorted in ascending order. If the value is found
     * in the buffer, the index of the value is returned. Otherwise, the index
     * of the first element in the buffer greater than the value is returned. If
     * the buffer is empty or all elements are less than the specified value, an
     * empty optional is returned.
     *
     * @param buffer The buffer to search in. Must be sorted in ascending order.
     * @param value The value to search for.
     * @return An optional index representing the lower bound of the value. If
     * the value is found, the index of the value is returned. Otherwise, the
     * index of the first element greater than the value is returned. If the
     * buffer is empty or all elements are less than the value, an empty
     * optional is returned.
     *
     * @sa Buffer
     */
    optional<dimn_t>
    lower_bound(const Buffer& buffer, ConstReference value) const;

    /**
     * @brief Find the upper bound of a value in a buffer.
     *
     * This method finds the position of the first element in the buffer that is
     * greater than the given value. The buffer is expected to be sorted in
     * non-decreasing order.
     *
     * @param buffer The buffer in which to search for the upper bound.
     * @param value The value for which to find the upper bound.
     * @return An optional containing the position of the upper bound if found,
     * or an empty optional if not found.
     * @sa Buffer
     */
    optional<dimn_t>
    upper_bound(const Buffer& buffer, ConstReference value) const;

    /**
     * @brief Finds the maximum value in the given buffer and stores it in the
     * provided output parameter.
     *
     * This method performs an algorithm to find the maximum value in the given
     * buffer. It uses the `buffer` parameter as input and calculates the
     * maximum value using an internally implemented algorithm. The maximum
     * value is then stored in the `out` parameter.
     *
     * Note: The `out` parameter must be a modifiable reference.
     *
     * @param buffer The input buffer to search for the maximum value.
     * @param out    The output parameter to store the maximum value.
     *
     * @sa Buffer
     */
    void max(const Buffer& buffer, Reference out) const;

    /**
     * @brief Find the minimum value in the given buffer and store the result in
     * the specified reference object.
     *
     * This method calculates the minimum value in the specified buffer by using
     * the MinFunctor algorithm. The result is then stored in the provided
     * reference object.
     *
     * @param buffer The buffer to find the minimum value from.
     * @param out The reference object to store the minimum value.
     *
     * @sa Buffer
     */
    void min(const Buffer& buffer, Reference out) const;

    /**
     * @brief Compare two buffers lexicographically.
     *
     * The comparison is performed by
     * iterating over the corresponding elements in both buffers and comparing
     * them in lexicographical order. The elements are compared using their
     * underlying value type's comparison operator.
     *
     * @param left The left buffer to compare.
     * @param right The right buffer to compare.
     * @return True if the left buffer is lexicographically less than the right
     * buffer, false otherwise.
     *
     * @sa Buffer
     */
    bool lexicographical_compare(const Buffer& left, const Buffer& right) const;
};

namespace algorithms {

RPY_NO_DISCARD inline optional<dimn_t>
find(const Buffer& buffer, ConstReference value)
{
    if (buffer.is_null()) { return {}; }
    auto& algo = buffer.device()->algorithms(buffer.type().get());
    return algo.find(buffer, value);
}

RPY_NO_DISCARD inline optional<dimn_t>
lower_bound(const Buffer& buffer, ConstReference value)
{
    if (buffer.is_null()) { return {}; }
    auto& algo = buffer.device()->algorithms(buffer.type().get());
    return algo.lower_bound(buffer, value);
}

RPY_NO_DISCARD inline optional<dimn_t>
upper_bound(const Buffer& buffer, ConstReference value)
{
    if (buffer.is_null()) { return {}; }
    auto& algo = buffer.device()->algorithms(buffer.type().get());
    return algo.lower_bound(buffer, value);
}

RPY_NO_DISCARD inline dimn_t count(const Buffer& buffer, ConstReference value)
{
    if (buffer.is_null()) { return 0; }

    auto& algo = buffer.device()->algorithms(buffer.type().get());
    return algo.count(buffer, value);
}

RPY_NO_DISCARD inline bool contains(const Buffer& buffer, ConstReference value)
{
    return static_cast<bool>(find(buffer, value));
}

RPY_NO_DISCARD inline optional<dimn_t>
mismatch(const Buffer& left, const Buffer& right)
{
    if (left.is_null()) {
        if (right.is_null()) { return {}; }
        return 0;
    }
    if (right.is_null()) { return 0; }

    Device devices[] = {left.device(), right.device()};
    const auto device = get_best_device(devices);

    auto& algo = device->algorithms(left.type().get(), right.type().get());
    return algo.mismatch(left, right);
}

RPY_NO_DISCARD inline bool equal(const Buffer& left, const Buffer& right)
{
    return !static_cast<bool>(mismatch(left, right));
}

RPY_NO_DISCARD inline bool
lexicographical_compare(const Buffer& left, const Buffer& right)
{
    if (left.is_null()) {
        if (right.is_null()) { return {}; }
        return false;
    }
    if (right.is_null()) { return false; }

    Device devices[] = {left.device(), right.device()};
    const auto device = get_best_device(devices);

    auto& algo = device->algorithms(left.type().get(), right.type().get());
    return algo.lexicographical_compare(left, right);
}

inline void copy(Buffer& dst, const Buffer& src)
{
    if (src.is_null()) { return; }

    Device device;
    const auto src_type = src.type();
    if (dst.is_null()) {
        device = src.device();
        dst = src_type->allocate(device, src.size());
    } else {
        Device devices[] = {dst.device(), src.device()};
        RPY_CHECK(src.size() <= dst.size());
        device = get_best_device(devices);
    }

    auto& algo = device->algorithms(dst.type().get(), src_type.get(), true);
    return algo.copy(dst, src);
}

inline void swap_ranges(Buffer& left, Buffer& right)
{
    if (left.is_null() && right.is_null()) { return; }
    RPY_CHECK(!left.is_null() && !right.is_null());

    RPY_CHECK(left.size() == right.size());
    const auto left_type = left.type();
    RPY_CHECK(left_type == right.type());

    Device devices[] = {left.device(), right.device()};
    Device device = get_best_device(devices);

    auto& algo = device->algorithms(left_type.get());
    algo.swap_ranges(left, right);
}

inline void reverse(Buffer& buffer)
{
    if (buffer.empty()) { return; }

    auto& algo = buffer.device()->algorithms(buffer.type().get());
    algo.reverse(buffer);
}

inline void shift_left(Buffer& buffer)
{
    if (buffer.empty()) { return; }

    auto& algo = buffer.device()->algorithms(buffer.type().get());
    algo.shift_left(buffer);
}

inline void shift_right(Buffer& buffer)
{
    if (buffer.empty()) { return; }

    auto& algo = buffer.device()->algorithms(buffer.type().get());
    algo.shift_left(buffer);
}

inline void fill(Buffer& buffer, ConstReference value)
{
    if (buffer.empty()) { return; }

    auto& algo = buffer.device()->algorithms(buffer.type().get());
    algo.fill(buffer, value);
}

inline void min(const Buffer& buffer, Reference out)
{
    if (buffer.empty()) { return; }

    auto& algo = buffer.device()->algorithms(buffer.type().get());
    algo.min(buffer, out);
}

inline void max(const Buffer& buffer, Reference out)
{
    if (buffer.empty()) { return; }

    auto& algo = buffer.device()->algorithms(buffer.type().get());
    algo.max(buffer, out);
}

}// namespace algorithms

template <template <typename...> class Implementor, typename... Ts>
void DeviceHandle::register_algorithm_drivers() const
{
    p_algorithms->template register_implementation<Implementor, Ts...>(
            get_type<Ts>()...
    );
}

template <template <typename...> class Implementor, typename... Ts>
void DeviceHandle::register_algorithm_drivers(dtl::TypePtrify<Ts>... types
) const
{
    p_algorithms->template register_implementation<Implementor, Ts...>(types...
    );
}

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICES_ALGORITHMS_H
