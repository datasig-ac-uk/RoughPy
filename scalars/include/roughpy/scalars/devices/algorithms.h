//
// Created by sam on 3/31/24.
//

#ifndef ROUGHPY_DEVICES_ALGORITHMS_H
#define ROUGHPY_DEVICES_ALGORITHMS_H

#include <roughpy/scalars/scalars_fwd.h>

#include "core.h"
#include "value.h"
#include "buffer.h"
#include "type.h"

#include <roughpy/core/types.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/alloc.h>

namespace rpy {
namespace devices {
namespace algorithms {
namespace drivers {


RPY_NO_DISCARD ROUGHPY_SCALARS_EXPORT optional<dimn_t>
find(const Type* type,
     const devices::Buffer& buffer,
     devices::Reference value);

RPY_NO_DISCARD inline bool contains(
        const Type* type,
        const devices::Buffer& buffer,
        devices::Reference value
)
{
    return static_cast<bool>(find(type, buffer, value));
}

RPY_NO_DISCARD ROUGHPY_SCALARS_EXPORT dimn_t
count(const Type* type,
      const devices::Buffer& buffer,
      devices::Reference value);

RPY_NO_DISCARD ROUGHPY_SCALARS_EXPORT optional<dimn_t> mismatch(
        const Type* left_type,
        const devices::Buffer& left,
        const Type* right_type,
        const devices::Buffer& right
);

RPY_NO_DISCARD inline bool
equal(const Type* left_type,
      const devices::Buffer& left,
      const Type* right_type,
      const devices::Buffer& right)
{
    return !static_cast<bool>(mismatch(left_type, left, right_type, right));
}

ROUGHPY_SCALARS_EXPORT void
swap(const Type* type, devices::Buffer& left, devices::Buffer& right);

ROUGHPY_SCALARS_EXPORT void
copy(const Type* type, devices::Buffer& dst, const devices::Buffer& src);

ROUGHPY_SCALARS_EXPORT void
fill(const Type* type, devices::Buffer& buffer, devices::Reference value);

ROUGHPY_SCALARS_EXPORT void
iota(const Type* type,
     devices::Buffer& buffer,
     devices::Reference start,
     devices::Reference stop,
     devices::Reference step);

ROUGHPY_SCALARS_EXPORT void
reverse(const Type* type, devices::Buffer& buffer);

ROUGHPY_SCALARS_EXPORT void
shift_left(const Type* type, devices::Buffer& buffer);

ROUGHPY_SCALARS_EXPORT void
shift_right(const Type* type, devices::Buffer& buffer);

ROUGHPY_SCALARS_EXPORT void lower_bound(
        devices::Reference out,
        const Type* type,
        const devices::Buffer& buffer,
        devices::Reference value
);

ROUGHPY_SCALARS_EXPORT void upper_bound(
        devices::Reference out,
        const Type* type,
        const devices::Buffer& buffer,
        devices::Reference value
);

ROUGHPY_SCALARS_EXPORT SliceIndex equal_range(
        const Type* type,
        const devices::Buffer& buffer,
        devices::Reference value
);

ROUGHPY_SCALARS_EXPORT bool binary_serach(
        const Type* type,
        const devices::Buffer& buffer,
        devices::Reference value
);

ROUGHPY_SCALARS_EXPORT void
max(devices::Reference out, const Type* type, const devices::Buffer& buffer
);

ROUGHPY_SCALARS_EXPORT void
min(devices::Reference out, const Type* type, const devices::Buffer& buffer
);

ROUGHPY_SCALARS_EXPORT bool lexicographical_compare(
        const Type* type,
        const devices::Buffer& left,
        const devices::Buffer& right
);


}
}// namespace algorithms
}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICES_ALGORITHMS_H
