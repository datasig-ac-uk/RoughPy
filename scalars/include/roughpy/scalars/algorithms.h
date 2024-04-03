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
namespace drivers = devices::algorithms::drivers;


inline const devices::Type* packed_to_type(const PackedScalarType& type)
{
    return type.is_pointer() ? type->as_type() : get_type(type.get_type_info());
}

template <typename T>
RPY_NO_DISCARD optional<dimn_t> find(const ScalarArray& range, const T& value)
{
    return drivers::find(
            packed_to_type(range.type()),
            range.buffer(),
            devices::TypedReference(value)
    );
}
RPY_NO_DISCARD inline optional<dimn_t>
find(const ScalarArray& range, const Scalar& value)
{
    return drivers::find(
            packed_to_type(range.type()),
            range.buffer(),
            devices::Reference(value.pointer())
    );
}

template <typename T>
RPY_NO_DISCARD bool contains(const ScalarArray& range, const T& value)
{
    return drivers::contains(
            packed_to_type(range.type()),
            range.buffer(),
            devices::TypedReference(value)
    );
}

RPY_NO_DISCARD inline bool
contains(const ScalarArray& range, const Scalar& value)
{
    return drivers::contains(
            packed_to_type(range.type()),
            range.buffer(),
            devices::Reference(value.pointer())
    );
}

template <typename T>
RPY_NO_DISCARD dimn_t count(const ScalarArray& range, const T& value)
{
    return drivers::count(
            packed_to_type(range.type()),
            range.buffer(),
            devices::TypedReference(value)
    );
}

RPY_NO_DISCARD inline dimn_t
count(const ScalarArray& range, const Scalar& value)
{
    return drivers::count(
            packed_to_type(range.type()),
            range.buffer(),
            devices::Reference(value.pointer())
    );
}

RPY_NO_DISCARD inline optional<dimn_t>
mismatch(const ScalarArray& left, const ScalarArray& right)
{
    return drivers::mismatch(
            left.type(),
            right.buffer(),
            left.type(),
            right.buffer()
    );
}

RPY_NO_DISCARD inline bool
equal(const ScalarArray& left, const ScalarArray& right)
{
    return drivers::equal(
            left.type(),
            left.buffer(),
            right.type(),
            right.buffer()
    );
}

inline void swap(ScalarArray& left, ScalarArray& right)
{
    RPY_CHECK(left.type() == right.type());
    drivers::swap(left.type(), left.mut_buffer(), right.mut_buffer());
}

inline void copy(ScalarArray& dst, const ScalarArray& src)
{
    drivers::copy(dst.type(), dst.mut_buffer(), src.buffer());
}

template <typename T>
inline void fill(ScalarArray& dst, const T& value)
{
    drivers::fill(dst.type(), dst.mut_buffer(), devices::TypedReference(value));
}

inline void fill(ScalarArray& dst, const Scalar& value)
{
    drivers::fill(
            dst.type(),
            dst.mut_buffer(),
            devices::Reference(value.pointer())
    );
}

}// namespace algorithms
}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_ALGORITHMS_H
