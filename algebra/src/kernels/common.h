//
// Created by sam on 15/04/24.
//

#ifndef COMMON_H
#define COMMON_H

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include <roughpy/devices/core.h>
#include <roughpy/scalars/scalar_array.h>
#include <roughpy/scalars/scalar_type.h>

#include "key_array.h"
#include "vector.h"

namespace rpy {
namespace algebra {
namespace dtl {

template <typename ArgType, typename Derived>
class ArgData;

struct MutableVectorArg;
struct ConstVectorArg;
struct ConstScalarArg;
struct MutableScalarArg;

template <typename... ArgSpec>
class GenericKernel;

}// namespace dtl
}// namespace algebra
}// namespace rpy

#endif// COMMON_H
