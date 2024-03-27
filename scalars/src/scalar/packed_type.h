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

constexpr PackedScalarType pack_type(devices::TypeInfo tinfo) noexcept
{
    return PackedScalarType(tinfo, {});
}
constexpr PackedScalarType pack_type(const ScalarType* type) noexcept
{
    return PackedScalarType(type, {});
}

template <typename E>
constexpr PackedScalarType pack_type(PackedScalarTypePointer<E> packed) noexcept
{
    return (packed.is_pointer()) ? PackedScalarType(packed.get_pointer(), {})
                                 : PackedScalarType(packed.get_type_info(), {});
}

}// namespace dtl
}// namespace scalars
}// namespace rpy
#endif// ROUGHPY_PACKED_TYPE_H
