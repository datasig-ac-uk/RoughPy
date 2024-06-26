//
// Created by sam on 24/06/24.
//

#ifndef ROUGHPY_SCALARS_SCALARS_FWD_H
#define ROUGHPY_SCALARS_SCALARS_FWD_H

#include <roughpy/core/macros.h>

#include <roughpy/devices/core.h>
#include <roughpy/devices/type.h>
#include <roughpy/devices/value.h>


#if RPY_HAS_INCLUDE("roughpy_algebra_export.h")
#  include "roughpy_algebra_export.h"
#elif RPY_HAS_INCLUDE(<roughpy/algebra/roughpy_platform_export.h>)
#  include <roughpy/algebra/roughpy_algebra_export.h>
#else
#define ROUGHPY_ALGEBRA_EXPORT
#define ROUGHPY_ALGEBRA_NO_EXPORT
#define ROUGHPY_ALGEBRA_DEPRECATED
#define ROUGHPY_ALGEBRA_DEPRECATED_EXPORT
#endif

#define ROUGHPY_SCALARS_EXPORT ROUGHPY_ALGEBRA_EXPORT
#define ROUGHPY_SCALARS_NO_EXPORT ROUGHPY_ALGEBRA_NO_EXPORT
#define ROUGHPY_SCALARS_DEPRECATED ROUGHPY_ALGEBRA_DEPRECATED
#define ROUGHPY_SCALARS_DEPRECATED_EXPORT ROUGHPY_ALGEBRA_DEPRECATED_EXPORT

#ifdef ROUGHPY_ALGEBRA_NO_DEPRECATED
#define ROUGHPY_SCALARS_NO_DEPRECATED
#endif


namespace rpy { namespace scalars {


using ScalarTypeCode = devices::TypeCode;

using seed_t = uint64_t;

using devices::get_type;
using devices::Type;
using devices::TypePtr;
namespace math = devices::math;

using Scalar = devices::Value;
using ScalarCRef = devices::ConstReference;
using ScalarRef = devices::Reference;

class ScalarArray;
class ScalarStream;
class ScalarVector;

class ScalarRandomGenerator;


}}

#endif //ROUGHPY_SCALARS_SCALARS_FWD_H
