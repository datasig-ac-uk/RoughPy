//
// Created by sam on 3/29/24.
//

#ifndef ROUGHPY_SCALARS_ALGORITHMS_H
#define ROUGHPY_SCALARS_ALGORITHMS_H

#include "scalars_fwd.h"

#include "devices/buffer.h"
#include "devices/core.h"
#include "devices/value.h"
#include "scalar.h"
#include "scalar_array.h"
#include "scalar_type.h"

namespace rpy {
namespace scalars {
namespace algorithms {
namespace drivers {

RPY_NO_DISCARD ROUGHPY_SCALARS_EXPORT optional<dimn_t>
find(PackedScalarType type,
     const devices::Buffer& buffer,
     devices::Reference value);

RPY_NO_DISCARD inline bool contains(
        PackedScalarType type,
        const devices::Buffer& buffer,
        devices::Reference value
)
{
    return static_cast<bool>(find(type, buffer, value));
}

RPY_NO_DISCARD ROUGHPY_SCALARS_EXPORT dimn_t
count(PackedScalarType type,
      const devices::Buffer& buffer,
      devices::Reference value);

RPY_NO_DISCARD ROUGHPY_SCALARS_EXPORT optional<dimn_t> mismatch(
        PackedScalarType left_type,
        const devices::Buffer& left,
        PackedScalarType right_type,
        const devices::Buffer& right
);

RPY_NO_DISCARD inline bool
equal(PackedScalarType left_type,
      const devices::Buffer& left,
      PackedScalarType right_type,
      const devices::Buffer& right)
{
    return !static_cast<bool>(mismatch(left_type, left, right_type, right));
}

ROUGHPY_SCALARS_EXPORT void
swap(PackedScalarType type, devices::Buffer& left, devices::Buffer& right);

ROUGHPY_SCALARS_EXPORT void
copy(PackedScalarType type, devices::Buffer& dst, const devices::Buffer& src);

ROUGHPY_SCALARS_EXPORT void
fill(PackedScalarType type, devices::Buffer& buffer, devices::Reference value);

ROUGHPY_SCALARS_EXPORT void
iota(PackedScalarType type,
     devices::Buffer& buffer,
     devices::Reference start,
     devices::Reference stop,
     devices::Reference step);

ROUGHPY_SCALARS_EXPORT void
reverse(PackedScalarType type, devices::Buffer& buffer);

ROUGHPY_SCALARS_EXPORT void
shift_left(PackedScalarType type, devices::Buffer& buffer);

ROUGHPY_SCALARS_EXPORT void
shift_right(PackedScalarType type, devices::Buffer& buffer);

ROUGHPY_SCALARS_EXPORT void lower_bound(
        devices::Reference out,
        PackedScalarType type,
        const devices::Buffer& buffer,
        devices::Reference value
);

ROUGHPY_SCALARS_EXPORT void upper_bound(
        devices::Reference out,
        PackedScalarType type,
        const devices::Buffer& buffer,
        devices::Reference value
);

ROUGHPY_SCALARS_EXPORT SliceIndex equal_range(
        PackedScalarType type,
        const devices::Buffer& buffer,
        devices::Reference value
);

ROUGHPY_SCALARS_EXPORT bool binary_serach(
        PackedScalarType type,
        const devices::Buffer& buffer,
        devices::Reference value
);

ROUGHPY_SCALARS_EXPORT void
max(devices::Reference out, PackedScalarType type, const devices::Buffer& buffer
);

ROUGHPY_SCALARS_EXPORT void
min(devices::Reference out, PackedScalarType type, const devices::Buffer& buffer
);

ROUGHPY_SCALARS_EXPORT bool lexicographical_compare(
        PackedScalarType type,
        const devices::Buffer& left,
        const devices::Buffer& right
);

}// namespace drivers

template <typename T>
RPY_NO_DISCARD optional<dimn_t> find(const ScalarArray& range, const T& value)
{
    return drivers::find(
            range.type(),
            range.buffer(),
            devices::TypedReference(value)
    );
}
RPY_NO_DISCARD inline optional<dimn_t>
find(const ScalarArray& range, const Scalar& value)
{
    return drivers::find(
            range.type(),
            range.buffer(),
            devices::Reference(value.pointer())
    );
}

template <typename T>
RPY_NO_DISCARD bool contains(const ScalarArray& range, const T& value)
{
    return drivers::contains(
            range.type(),
            range.buffer(),
            devices::TypedReference(value)
    );
}

RPY_NO_DISCARD inline bool
contains(const ScalarArray& range, const Scalar& value)
{
    return drivers::contains(
            range.type(),
            range.buffer(),
            devices::Reference(value.pointer())
    );
}

template <typename T>
RPY_NO_DISCARD dimn_t count(const ScalarArray& range, const T& value)
{
    return drivers::count(
            range.type(),
            range.buffer(),
            devices::TypedReference(value)
    );
}

RPY_NO_DISCARD inline dimn_t
count(const ScalarArray& range, const Scalar& value)
{
    return drivers::count(
            range.type(),
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
