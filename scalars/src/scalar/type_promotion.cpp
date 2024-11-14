//
// Created by sam on 1/19/24.
//

#include "type_promotion.h"

#include "roughpy/core/check.h"            // for throw_exception, RPY_THROW
#include "roughpy/core/debug_assertion.h"  // for RPY_DBG_ASSERT

#include <roughpy/scalars/traits.h>

using namespace rpy;
using namespace rpy::scalars;

using scalars::dtl::ScalarContentType;

namespace {

devices::TypeInfo compute_int_promotion(
        const devices::TypeInfo& dst_info,
        const devices::TypeInfo& src_info
)
{
    if (traits::is_integral(src_info)) {
        devices::TypeInfo info{
                dst_info.code,
                std::max(dst_info.bytes, src_info.bytes),
                std::max(dst_info.alignment, src_info.alignment),
                1
        };

        if (src_info.code != info.code) {
            // One is signed, the other is unsigned
            if (traits::is_signed(src_info)) {
                // If src is signed the we must make sure that the dst is signed
                info.code = src_info.code;
            } else {
                // If src is unsigned then dst might not be large enough
                if (info.bytes <= src_info.bytes
                    && info.bytes < sizeof(int64_t)) {
                    info.bytes *= 2;
                    info.alignment = info.bytes;
                }
            }
        }

        return info;
    }

    // In almost all other circumstances, the correct promotion is to simply use
    // the src type.
    return src_info;
}

devices::TypeInfo compute_float_promotion(
        const devices::TypeInfo& dst_info,
        const devices::TypeInfo& src_info
)
{
    // Floats should promote to anything that isn't an int
    if (traits::is_integral(src_info)) {
        // TODO: Make sure dst is large enough
        return dst_info;
    }

    if (traits::is_floating_point(src_info)) {
        if (src_info.bytes > dst_info.bytes) {
            return {devices::TypeCode::Float,
                    src_info.bytes,
                    src_info.alignment,
                    1};
        }

        return dst_info;
    }

    return src_info;
}

devices::TypeInfo compute_bfloat_promotion(
        const devices::TypeInfo& dst_info,
        const devices::TypeInfo& src_info
)
{
    if (traits::is_integral(src_info)) {
        /*
         * bfloats cannot hold anything larger than int8_t faithfully, so for
         * anything larger than int8_t we'll promote to a float or double.
         */
        if (src_info.bytes == 1) { return dst_info; }
        if (src_info.bytes <= 4) { return devices::type_info<float>(); }

        return devices::type_info<double>();
    }

    return src_info;
}

devices::TypeInfo compute_aprational_promotion(
        const devices::TypeInfo& dst_info,
        const devices::TypeInfo& src_info
)
{
    // Rationals only promote to rational polys
    if (src_info.code == devices::TypeCode::APRationalPolynomial) {
        return src_info;
    }

    return dst_info;
}

devices::TypeInfo compute_aprpoly_promotion(
        const devices::TypeInfo& dst_info,
        const devices::TypeInfo& RPY_UNUSED_VAR src_info
)
{
    return dst_info;
}

devices::TypeInfo compute_promotion(
        PackedScalarTypePointer<ScalarContentType> dst_type,
        PackedScalarTypePointer<ScalarContentType> src_type
)
{
    const auto dst_info = type_info_from(dst_type);
    const auto src_info = type_info_from(src_type);

    return compute_type_promotion(src_info, dst_info);
}

}// namespace

devices::TypeInfo
scalars::compute_type_promotion(devices::TypeInfo left, devices::TypeInfo right)
{
    if (left == right) { return left; }
    switch (right.code) {
        case devices::TypeCode::Int:
        case devices::TypeCode::UInt: return compute_int_promotion(right, left);
        case devices::TypeCode::Float:
            return compute_float_promotion(right, left);
        case devices::TypeCode::BFloat:
            return compute_bfloat_promotion(right, left);
        case devices::TypeCode::ArbitraryPrecisionRational:
            return compute_aprational_promotion(right, left);
        case devices::TypeCode::APRationalPolynomial:
            return compute_aprpoly_promotion(right, left);
        case devices::TypeCode::Bool:
        case devices::TypeCode::OpaqueHandle:
        case devices::TypeCode::Complex:
        case devices::TypeCode::Rational:
        case devices::TypeCode::ArbitraryPrecisionInt:
        case devices::TypeCode::ArbitraryPrecisionUInt:
        case devices::TypeCode::ArbitraryPrecisionFloat:
        case devices::TypeCode::ArbitraryPrecisionComplex: break;
    }
    RPY_THROW(std::runtime_error, "cannot find suitable promotion");
}


devices::TypeInfo scalars::dtl::compute_dest_type(
        PackedScalarTypePointer<ScalarContentType> dst_type,
        PackedScalarTypePointer<ScalarContentType> src_type
)
{
    RPY_DBG_ASSERT(!(dst_type == src_type));

    switch (dst_type.get_enumeration()) {
        case ScalarContentType::TrivialBytes:
        case ScalarContentType::ConstTrivialBytes:
        case ScalarContentType::OwnedPointer:
            return compute_promotion(dst_type, src_type);
        case ScalarContentType::OpaquePointer:
        case ScalarContentType::ConstOpaquePointer:
        case ScalarContentType::Interface:
        case ScalarContentType::OwnedInterface: return dst_type;
    }

    RPY_UNREACHABLE_RETURN(dst_type);
}
