//
// Created by sam on 3/29/24.
//

#ifndef ROUGHPY_SCALARS_ALGORITHMS_H
#define ROUGHPY_SCALARS_ALGORITHMS_H

#include "scalars_fwd.h"

#include "scalar.h"
#include "scalar_array.h"
#include "scalar_type.h"
#include <roughpy/devices/algorithms.h>
#include <roughpy/devices/buffer.h>
#include <roughpy/devices/core.h>
#include <roughpy/devices/value.h>

namespace rpy {
namespace scalars {
namespace algorithms {
namespace drivers = devices::algorithms;

/**
 * @brief Find the first occurrence of a given value in a ScalarArray range.
 *
 * @param range The ScalarArray range to search in.
 * @param value The value to search for.
 *
 * @return An optional containing the index of the first occurrence of the
 * value, if found. If the value is not found, the optional will be empty.
 */
RPY_NO_DISCARD inline optional<dimn_t>
find(const ScalarArray& range, const Scalar& value)
{
    return drivers::find(
            range.buffer(),
            devices::ConstReference(value.pointer(), value.type()->as_type())
    );
}

/**
 * @brief Find the first occurrence of a given value in a ScalarArray range.
 *
 * @param range The ScalarArray range to search in.
 * @param value The value to search for.
 *
 * @return An optional containing the index of the first occurrence of the
 * value, if found. If the value is not found, the optional will be empty.
 */
template <typename T>
RPY_NO_DISCARD optional<dimn_t> find(const ScalarArray& range, const T& value)
{
    return find(range, Scalar(range.type(), &value));
}

/**
 * @brief Find the lower bound of a given value in a ScalarArray range.
 *
 * This method returns an optional containing the index of the first element in
 * the range that is not less than the given value. If no such element is found,
 * the optional will be empty.
 *
 * @param range The ScalarArray range to search in.
 * @param value The value to find the lower bound for.
 *
 * @return An optional containing the index of the lower bound, if found. If no
 * lower bound is found, the optional will be empty.
 */
RPY_NO_DISCARD inline optional<dimn_t>
lower_bound(const ScalarArray& range, const Scalar& value)
{
    return drivers::lower_bound(
            range.buffer(),
            devices::ConstReference(value.pointer(), value.type()->as_type())
    );
}

/**
 * @brief Find the first element in the ScalarArray range that is not less than
 * the given value.
 *
 * @param range The ScalarArray range to search in.
 * @param value The value to search for. The type must be compatible with the
 * ScalarArray.
 *
 * @return An optional containing the index of the first element that is not
 * less than the value, if found. If the value is not found or the range is
 * empty, the optional will be empty.
 */
template <typename T>
RPY_NO_DISCARD optional<dimn_t>
lower_bound(const ScalarArray& range, const T& value)
{
    return lower_bound(range, Scalar(range.type(), &value));
}

/**
 * @brief Find the upper bound of a given value in a ScalarArray range.
 *
 * This function returns an optional that contains the index of the
 * upper_bound of the value in the range if it exists. If the value
 * is not found or if the upper bound cannot be determined, the optional
 * will be empty.
 *
 * @param range The ScalarArray range to search in.
 * @param value The value to find the upper bound of.
 *
 * @return An optional containing the index of the upper bound of the value,
 * if found. If the value is not found or if the upper bound cannot be
 * determined, the optional will be empty.
 */
RPY_NO_DISCARD inline optional<dimn_t>
upper_bound(const ScalarArray& range, const Scalar& value)
{
    return drivers::upper_bound(
            range.buffer(),
            devices::ConstReference(value.pointer(), value.type()->as_type())
    );
}

/**
 * @brief Find the upper bound of the first occurrence of a given value in a
 * ScalarArray range.
 *
 * The upper bound is defined as the first element that is greater than the
 * given value. If the given value is greater than or equal to all elements in
 * the range, the upper bound will be the end of the range.
 *
 * @param range The ScalarArray range to search in.
 * @param value The value to search for.
 *
 * @return An optional containing the index of the upper bound, if found. If the
 * value is not found, the optional will be empty.
 */
template <typename T>
RPY_NO_DISCARD optional<dimn_t>
upper_bound(const ScalarArray& range, const T& value)
{
    return upper_bound(range, Scalar(range.type(), &value));
}

/**
 * @brief Count the number of occurrences of a given value in a ScalarArray
 * range.
 *
 * @param range The ScalarArray range to search in.
 * @param value The value to count occurrences for.
 *
 * @return The number of occurrences of the value in the range.
 */
RPY_NO_DISCARD inline dimn_t
count(const ScalarArray& range, const Scalar& value)
{
    return drivers::count(
            range.buffer(),
            devices::ConstReference(value.pointer(), value.type()->as_type())
    );
}

/**
 * @brief Count the occurrences of a given value in a ScalarArray range.
 *
 * @param range The ScalarArray range to count in.
 * @param value The value to count.
 *
 * @return The number of occurrences of the value in the range.
 */
template <typename T>
RPY_NO_DISCARD dimn_t count(const ScalarArray& range, const T& value)
{
    return count(range, Scalar(range.type(), &value));
}

/**
 * @brief Check if a given value is contained within a ScalarArray range.
 *
 * @param range The ScalarArray range to check.
 * @param value The value to search for.
 *
 * @return True if the value is found in the range, false otherwise.
 */
RPY_NO_DISCARD inline bool
contains(const ScalarArray& range, const Scalar& value)
{
    return drivers::contains(
            range.buffer(),
            devices::ConstReference(value.pointer(), value.type()->as_type())
    );
}

/**
 * @brief Checks if a given value is contained in a ScalarArray range.
 *
 * @param range The ScalarArray range to check in.
 * @param value The value to check for.
 *
 * @return True if the value is found in the range, false otherwise.
 */
template <typename T>
RPY_NO_DISCARD bool contains(const ScalarArray& range, const T& value)
{
    return contains(range, Scalar(range.type(), &value));
}

/**
 * @brief Find the first mismatch between two ScalarArrays.
 *
 * This function compares the elements of the two ScalarArrays sequentially
 * and returns an optional containing the index of the first mismatch if found,
 * or an empty optional if the two ScalarArrays are equal.
 *
 * @param left The first ScalarArray to compare.
 * @param right The second ScalarArray to compare.
 *
 * @return An optional containing the index of the first mismatch, if found.
 * If the two ScalarArrays are equal, the optional will be empty.
 */
RPY_NO_DISCARD inline optional<dimn_t>
mismatch(const ScalarArray& left, const ScalarArray& right)
{
    return drivers::mismatch(left.buffer(), right.buffer());
}

/**
 * @brief Check if two ScalarArray objects are equal.
 *
 * The equal method checks whether two ScalarArray objects have the
 * same contents by comparing their underlying buffer.
 *
 * @param left The first ScalarArray object to compare.
 * @param right The second ScalarArray object to compare.
 *
 * @return true if the contents of the two ScalarArray objects are equal,
 * false otherwise.
 */
RPY_NO_DISCARD inline bool
equal(const ScalarArray& left, const ScalarArray& right)
{
    return drivers::equal(left.buffer(), right.buffer());
}

/**
 * @brief Swaps the elements between two ScalarArray ranges.
 *
 * This function swaps the elements between two ScalarArray ranges. The ranges
 * are specified by two ScalarArray references, `left` and `right`. The elements
 * within the ranges are swapped using the `swap_ranges` function defined in the
 * `drivers` namespace. Before swapping, this function checks if both `left` and
 * `right` ScalarArray objects have the same type using the `type` function. If
 * they don't have the same type, an exception is thrown.
 *
 * @param left The ScalarArray range to swap elements from.
 * @param right The ScalarArray range to swap elements to.
 *
 * @see drivers::swap_ranges()
 *
 * @note This function assumes that `left` and `right` ScalarArray objects are
 *       mutable.
 */
inline void swap_ranges(ScalarArray& left, ScalarArray& right)
{
    RPY_CHECK(left.type() == right.type());
    drivers::swap_ranges(left.mut_buffer(), right.mut_buffer());
}

/**
 * @brief Copy the contents of one ScalarArray to another.
 *
 * This method copies the contents of the source ScalarArray to the destination
 * ScalarArray. The contents are copied element by element from the source to
 * the destination, overwriting the previous contents of the destination.
 *
 * @param dst The destination ScalarArray where the contents will be copied to.
 * @param src The source ScalarArray from which the contents will be copied.
 */
inline void copy(ScalarArray& dst, const ScalarArray& src)
{
    drivers::copy(dst.mut_buffer(), src.buffer());
}

/**
 * @brief Reverse the elements in a ScalarArray.
 *
 * This method reverses the order of the elements in the provided ScalarArray.
 *
 * @param arr The ScalarArray to be reversed.
 */
inline void reverse(ScalarArray& arr) { drivers::reverse(arr.mut_buffer()); }

/**
 * @brief Shifts the elements of a ScalarArray to the left by a given count.
 *
 * @param arr The ScalarArray to shift.
 * @param count The number of positions to shift the elements to the left.
 */
inline void shift_left(ScalarArray& arr, dimn_t count)
{
    (void) count;
    drivers::shift_left(arr.mut_buffer());
}

/**
 * @brief Shifts the elements of a ScalarArray to the right by a specified
 * count.
 *
 * This function shifts the elements of the ScalarArray to the right by the
 * specified count.
 *
 * @param arr The ScalarArray whose elements are to be shifted.
 * @param count The number of positions to shift the elements to the right.
 *        This parameter is ignored in the current implementation.
 *
 */
inline void shift_right(ScalarArray& arr, dimn_t count)
{
    (void) count;
    drivers::shift_right(arr.mut_buffer());
}

/**
 * @brief Fill the ScalarArray with a specified value.
 *
 * This method fills the specified ScalarArray with the provided value.
 *
 * @param dst The ScalarArray to fill.
 * @param value The value to fill the ScalarArray with.
 *
 * @note The ScalarArray will be modified in-place.
 */
inline void fill(ScalarArray& dst, const Scalar& value)
{
    drivers::fill(
            dst.mut_buffer(),
            devices::ConstReference(value.pointer(), value.type()->as_type())
    );
}

/**
 @brief Fill a ScalarArray with a given value.

 This method fills a ScalarArray with a specified value. The value is passed as
 a reference to a constant T object.

 @param dst The ScalarArray to be filled.
 @param value The value to fill the ScalarArray with.

 @see ScalarArray

 @code
 // Example usage:
 ScalarArray arr;
 fill(arr, 5);
 @endcode
 */
template <typename T>
inline void fill(ScalarArray& dst, const T& value)
{
    fill(dst, Scalar(dst.type(), &value));
}

/**
 * @brief Lexicographically compares two ScalarArrays.
 *
 * @param left The first ScalarArray to be compared.
 * @param right The second ScalarArray to be compared.
 *
 * @return True if the first ScalarArray is less than the second ScalarArray
 *         lexicographically, false otherwise.
 */
RPY_NO_DISCARD inline bool
lexicographical_compare(const ScalarArray& left, const ScalarArray& right)
{
    return drivers::lexicographical_compare(left.buffer(), right.buffer());
}

/**
 * @brief Find the minimum value in a given ScalarArray range.
 *
 * @param range The ScalarArray range to search in.
 *
 * @return The minimum value found in the range.
 */
RPY_NO_DISCARD inline Scalar min(const ScalarArray& range)
{
    Scalar result(range.type());
    drivers::min(
            range.buffer(),
            devices::Reference(result.mut_pointer(), result.type()->as_type())
    );
    return result;
}

/**
 * @brief Calculate the maximum value of a ScalarArray range.
 *
 * This method calculates the maximum value within the given ScalarArray range.
 * It iterates through each element in the range and compares it against the
 * current maximum value. The maximum value is then returned.
 *
 * @param range The ScalarArray range to calculate the maximum value from.
 *
 * @return The maximum value within the ScalarArray range.
 */
RPY_NO_DISCARD inline Scalar max(const ScalarArray& range)
{
    Scalar result(range.type());
    drivers::max(
            range.buffer(),
            devices::Reference(result.mut_pointer(), result.type()->as_type())
    );
    return result;
}

}// namespace algorithms
}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_ALGORITHMS_H
