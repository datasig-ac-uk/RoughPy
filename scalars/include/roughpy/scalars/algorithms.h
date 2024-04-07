//
// Created by sam on 3/29/24.
//

#ifndef ROUGHPY_SCALARS_ALGORITHMS_H
#define ROUGHPY_SCALARS_ALGORITHMS_H

#include "scalars_fwd.h"

#include "devices/algorithms.h"
#include "devices/buffer.h"
#include "devices/core.h"
#include "devices/value.h"
#include "scalar.h"
#include "scalar_array.h"
#include "scalar_type.h"

namespace rpy {
namespace scalars {
namespace algorithms {
namespace drivers = devices::algorithms;

RPY_NO_DISCARD inline optional<dimn_t>
find(const ScalarArray& range, const Scalar& value)
{
    return drivers::find(range.buffer(), devices::Reference(value.pointer()));
}

template <typename T>
RPY_NO_DISCARD optional<dimn_t> find(const ScalarArray& range, const T& value)
{
    return find(range.buffer(), Scalar(range.type(), &value));
    ;
}



RPY_NO_DISCARD inline optional<dimn_t>
lower_bound(const ScalarArray& range, const Scalar& value)
{
    return drivers::lower_bound(
            range.buffer(),
            devices::Reference(value.pointer())
    );
}

template <typename T>
RPY_NO_DISCARD optional<dimn_t>
lower_bound(const ScalarArray& range, const T& value)
{
    return lower_bound(range.buffer(), Scalar(range.type(), &value));
}

RPY_NO_DISCARD inline optional<dimn_t>
upper_bound(const ScalarArray& range, const Scalar& value)
{
    return drivers::upper_bound(
            range.buffer(),
            devices::Reference(value.pointer())
    );
}

template <typename T>
RPY_NO_DISCARD optional<dimn_t>
upper_bound(const ScalarArray& range, const T& value)
{
    return upper_bound(range.buffer(), Scalar(range.type(), &value));
}

RPY_NO_DISCARD inline dimn_t
count(const ScalarArray& range, const Scalar& value)
{
    return drivers::count(range.buffer(), devices::Reference(value.pointer()));
}

template <typename T>
RPY_NO_DISCARD dimn_t count(const ScalarArray& range, const T& value)
{
    return count(range.buffer(), Scalar(range.type(), &value));
}

RPY_NO_DISCARD inline bool
contains(const ScalarArray& range, const Scalar& value)
{
    return drivers::contains(
            range.buffer(),
            devices::Reference(value.pointer())
    );
}

template <typename T>
RPY_NO_DISCARD bool contains(const ScalarArray& range, const T& value)
{
    return contains(range.buffer(), Scalar(range.type(), &value));
}


RPY_NO_DISCARD inline optional<dimn_t>
mismatch(const ScalarArray& left, const ScalarArray& right)
{
    return drivers::mismatch(left.buffer(), right.buffer());
}

RPY_NO_DISCARD inline bool
equal(const ScalarArray& left, const ScalarArray& right)
{
    return drivers::equal(left.buffer(), right.buffer());
}

inline void swap_ranges(ScalarArray& left, ScalarArray& right)
{
    RPY_CHECK(left.type() == right.type());
    drivers::swap_ranges(left.mut_buffer(), right.mut_buffer());
}

inline void copy(ScalarArray& dst, const ScalarArray& src)
{
    drivers::copy(dst.mut_buffer(), src.buffer());
}

inline void reverse(ScalarArray& arr) { drivers::reverse(arr.mut_buffer()); }

inline void shift_left(ScalarArray& arr, dimn_t count)
{
    (void) count;
    drivers::shift_left(arr.mut_buffer());
}

inline void shift_right(ScalarArray& arr, dimn_t count)
{
    (void) count;
    drivers::shift_right(arr.mut_buffer());
}

inline void fill(ScalarArray& dst, const Scalar& value)
{
    drivers::fill(dst.mut_buffer(), devices::Reference(value.pointer()));
}

template <typename T>
inline void fill(ScalarArray& dst, const T& value)
{
    fill(dst, Scalar(dst.type(), &value));
}

RPY_NO_DISCARD inline bool
lexicographical_compare(const ScalarArray& left, const ScalarArray& right)
{
    return drivers::lexicographical_compare(left.buffer(), right.buffer());
}

RPY_NO_DISCARD inline Scalar min(const ScalarArray& range)
{
    Scalar result(range.type());
    drivers::min(range.buffer(), devices::Reference(result.mut_pointer()));
    return result;
}

RPY_NO_DISCARD inline Scalar max(const ScalarArray& range)
{
    Scalar result(range.type());
    drivers::max(range.buffer(), devices::Reference(result.mut_pointer()));
    return result;
}

}// namespace algorithms
}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_ALGORITHMS_H
