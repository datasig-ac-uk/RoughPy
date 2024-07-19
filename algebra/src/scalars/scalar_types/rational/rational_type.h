//
// Created by sam on 24/06/24.
//

#ifndef ROUGHPY_SCALAR_IMPLEMENTATIONS_RATIONAL_TYPE_H
#define ROUGHPY_SCALAR_IMPLEMENTATIONS_RATIONAL_TYPE_H

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>
#include <roughpy/devices/type.h>
#include <roughpy/devices/buffer.h>
#include <roughpy/devices/value.h>
#include <roughpy/devices/device_handle.h>

#include "rational.h"

namespace rpy {
namespace scalars {
namespace implementations {

template <typename Integral>
class RPY_LOCAL RationalType : public devices::Type {
    using value_type = Rational<Integral>;


public:
    RationalType();
};

extern template class RationalType<int32_t>;
extern template class RationalType<int64_t>;


using Rational32Type = RationalType<int32_t>;
using Rational64Type = RationalType<int64_t>;


} // implementations
} // scalars
} // rpy

#endif //ROUGHPY_SCALAR_IMPLEMENTATIONS_RATIONAL_TYPE_H
