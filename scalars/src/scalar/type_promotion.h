//
// Created by sam on 1/19/24.
//

#ifndef ROUGHPY_TYPE_PROMOTION_H
#define ROUGHPY_TYPE_PROMOTION_H

#include "packed_type.h"
#include <roughpy/scalars/scalar.h>

namespace rpy {
namespace scalars {
namespace dtl {

RPY_LOCAL
devices::TypeInfo compute_promotion(PackedType dst_type, PackedType src_type);

devices::TypeInfo compute_dest_type(
        PackedScalarTypePointer<ScalarContentType> dst_type,
        PackedScalarTypePointer<ScalarContentType> src_type
);

template <typename D, typename S>
devices::TypeInfo compute_dest_type(D dst_type, S src_type)
{
    return compute_promotion(pack_type(dst_type), pack_type(src_type));
}

}// namespace dtl
}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_TYPE_PROMOTION_H
