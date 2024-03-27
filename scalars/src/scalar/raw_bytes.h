//
// Created by sam on 14/11/23.
//

#ifndef ROUGHPY_SCALARS_RAW_BYTES_H
#define ROUGHPY_SCALARS_RAW_BYTES_H

#include "scalars_fwd.h"
#include <roughpy/core/types.h>

#include <vector>

namespace rpy {
namespace scalars {
namespace dtl {

RPY_NO_DISCARD std::vector<byte>
to_raw_bytes(const void* ptr, dimn_t size, PackedScalarType info);

void from_raw_bytes(
        void* dst,
        dimn_t count,
        Slice<byte> bytes,
        PackedScalarType info
);

}// namespace dtl
}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_RAW_BYTES_H
