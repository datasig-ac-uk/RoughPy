//
// Created by sam on 1/19/24.
//

#ifndef ROUGHPY_TYPE_PROMOTION_H
#define ROUGHPY_TYPE_PROMOTION_H

#include <roughpy/scalars/scalar.h>

namespace rpy {
namespace scalars {
namespace dtl {

devices::TypeInfo compute_dest_type(
        PackedScalarTypePointer<ScalarContentType> dst_type,
        PackedScalarTypePointer<ScalarContentType> src_type
);

}
}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_TYPE_PROMOTION_H
