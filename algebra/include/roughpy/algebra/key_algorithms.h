//
// Created by sam on 08/04/24.
//

#ifndef ROUGHPY_ALGEBRA_KEY_ALGORITHMS_H
#define ROUGHPY_ALGEBRA_KEY_ALGORITHMS_H

#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>

#include <roughpy/devices/algorithms.h>
#include <roughpy/devices/device_handle.h>

#include "key_array.h"
#include "key_scalar_array.h"

namespace rpy {
namespace algebra {
namespace algorithms {

namespace drivers = devices::algorithms;

RPY_NO_DISCARD inline optional<dimn_t>
find(const KeyArray& range, const BasisKey& value)
{
    return drivers::find(range.buffer(), devices::TypedReference(value));
}

RPY_NO_DISCARD inline dimn_t count(const KeyArray& range, const BasisKey& value)
{
    return drivers::count(range.buffer(), devices::TypedReference(value));
}

RPY_NO_DISCARD inline optional<dimn_t>
lower_bound(const KeyArray& range, const BasisKey& value)
{
    return drivers::lower_bound(range.buffer(), devices::TypedReference(value));
}

RPY_NO_DISCARD inline optional<dimn_t>
upper_bound(const KeyArray& range, const BasisKey& value)
{
    return drivers::upper_bound(range.buffer(), devices::TypedReference(value));
}

RPY_NO_DISCARD inline bool
contains(const KeyArray& range, const BasisKey& value)
{
    return static_cast<bool>(
            drivers::find(range.buffer(), devices::TypedReference(value))
    );
}

inline void fill(KeyArray& range, const BasisKey& value)
{
    drivers::fill(range.mut_buffer(), devices::TypedReference(value));
}

// mismatch
RPY_NO_DISCARD inline optional<dimn_t>
mismatch(const KeyArray& range1, const KeyArray& range2)
{
    return drivers::mismatch(range1.buffer(), range2.buffer());
}

// equal
RPY_NO_DISCARD inline bool equal(const KeyArray& range1, const KeyArray& range2)
{
    return drivers::equal(range1.buffer(), range2.buffer());
}

// swap_ranges
inline void swap_ranges(KeyArray& range1, KeyArray& range2)
{
    drivers::swap_ranges(range1.mut_buffer(), range2.mut_buffer());
}

// copy
inline void copy(KeyArray& dest, const KeyArray& src)
{
    drivers::copy(dest.mut_buffer(), src.buffer());
}

// reverse
inline void reverse(KeyArray& range) { drivers::reverse(range.mut_buffer()); }

// shift_left
inline void shift_left(KeyArray& range, dimn_t n)
{
    (void) n;
    drivers::shift_left(range.mut_buffer());
}

// shift_right
inline void shift_right(KeyArray& range, dimn_t n)
{
    (void) n;
    drivers::shift_right(range.mut_buffer());
}

RPY_NO_DISCARD inline bool
lexicographical_compare(const KeyArray& left, const KeyArray& right)
{
    return drivers::lexicographical_compare(left.buffer(), right.buffer());
}

RPY_NO_DISCARD inline BasisKey min(const KeyArray& range)
{
    BasisKey result;
    drivers::min(range.buffer(), devices::TypedReference(result));
    return result;
}

RPY_NO_DISCARD inline BasisKey max(const KeyArray& range)
{
    BasisKey result;
    drivers::max(range.buffer(), devices::TypedReference(result));
    return result;
}

}// namespace algorithms
}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_KEY_ALGORITHMS_H
