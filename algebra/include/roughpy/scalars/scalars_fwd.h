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
#elif RPY_HAS_INCLUDE(<roughpy / algebra / roughpy_platform_export.h>)
#  include <roughpy/algebra/roughpy_algebra_export.h>
#else
#  define ROUGHPY_ALGEBRA_EXPORT
#  define ROUGHPY_ALGEBRA_NO_EXPORT
#  define ROUGHPY_ALGEBRA_DEPRECATED
#  define ROUGHPY_ALGEBRA_DEPRECATED_EXPORT
#endif

#define ROUGHPY_SCALARS_EXPORT ROUGHPY_ALGEBRA_EXPORT
#define ROUGHPY_SCALARS_NO_EXPORT ROUGHPY_ALGEBRA_NO_EXPORT
#define ROUGHPY_SCALARS_DEPRECATED ROUGHPY_ALGEBRA_DEPRECATED
#define ROUGHPY_SCALARS_DEPRECATED_EXPORT ROUGHPY_ALGEBRA_DEPRECATED_EXPORT

#ifdef ROUGHPY_ALGEBRA_NO_DEPRECATED
#  define ROUGHPY_SCALARS_NO_DEPRECATED
#endif

namespace rpy {
namespace scalars {

using ScalarTypeCode = devices::TypeCode;

using seed_t = uint64_t;

using devices::get_type;
using devices::Type;
using devices::TypePtr;
namespace math = devices::math;

using Scalar = devices::Value;
using ScalarCRef = devices::ConstReference;
using ScalarRef = devices::Reference;

using devices::value_cast;

class ScalarArray;
class ScalarStream;
class ScalarVector;

class ScalarRandomGenerator;

class ROUGHPY_SCALARS_EXPORT BuiltinTypes
{
    TypePtr int8;
    TypePtr int16;
    TypePtr int32;
    TypePtr int64;

    TypePtr uint8;
    TypePtr uint16;
    TypePtr uint32;
    TypePtr uint64;

    TypePtr float16;
    TypePtr float32;
    TypePtr float64;

    TypePtr bfloat16;

    TypePtr rational;

    TypePtr polynomial;

public:
    BuiltinTypes();

    TypePtr get_int8() const noexcept { return int8; }
    TypePtr get_int16() const noexcept { return int16; }
    TypePtr get_int32() const noexcept { return int32; }
    TypePtr get_int64() const noexcept { return int64; }

    TypePtr get_uint8() const noexcept { return uint8; }
    TypePtr get_uint16() const noexcept { return int16; }
    TypePtr get_uint32() const noexcept { return int32; }
    TypePtr get_uint64() const noexcept { return int64; }

    TypePtr get_float16() const noexcept { return float16; }
    TypePtr get_float32() const noexcept { return float32; }
    TypePtr get_float64() const noexcept { return float64; }

    TypePtr get_bfloat16() const noexcept { return bfloat16; }

    TypePtr get_rational() const noexcept { return rational; }
    TypePtr get_polynomial() const noexcept { return polynomial; }

    TypePtr get_int(dimn_t bits) const;
    TypePtr get_uint(dimn_t bits) const;
    TypePtr get_rational(dimn_t bits) const;
};

extern ROUGHPY_SCALARS_EXPORT const BuiltinTypes builtin_types;

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SCALARS_FWD_H
