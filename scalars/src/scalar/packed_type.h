//
// Created by sam on 3/12/24.
//

#ifndef ROUGHPY_PACKED_TYPE_H
#define ROUGHPY_PACKED_TYPE_H

#include "packed_scalar_type_ptr.h"
#include "scalar.h"

namespace rpy {
namespace scalars {
namespace dtl {

enum EmptyEnum
{
};

using PackedType = PackedScalarTypePointer<EmptyEnum>;

constexpr PackedType pack_type(devices::TypeInfo tinfo) noexcept
{
    return PackedType(tinfo, {});
}
constexpr PackedType pack_type(const ScalarType* type) noexcept
{
    return PackedType(type, {});
}

template <typename E>
constexpr PackedType pack_type(PackedScalarTypePointer<E> packed) noexcept
{
    return (packed.is_pointer()) ? PackedType(packed.get_pointer(), {})
                                 : PackedType(packed.get_type_info(), {});
}

}// namespace dtl
}// namespace scalars
}// namespace rpy
#endif// ROUGHPY_PACKED_TYPE_H
