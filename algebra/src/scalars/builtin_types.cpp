//
// Created by sam on 8/8/24.
//

#include <roughpy/core/types.h>

#include <roughpy/devices/core.h>
#include <roughpy/devices/type.h>

#include "scalar_types/bfloat16/bfloat_16_type.h"
#include "scalar_types/half/half_type.h"
#include "scalar_types/polynomial/poly_rational_type.h"
#include "scalar_types/rational/arbitrary_precision_rational_type.h"
#include "scalars_fwd.h"

using namespace rpy;
using namespace rpy::devices;
using namespace rpy::scalars;

BuiltinTypes::BuiltinTypes()
{
    int8 = devices::get_type<int8_t>();
    int16 = devices::get_type<int16_t>();
    int32 = devices::get_type<int32_t>();
    int64 = devices::get_type<int64_t>();

    uint8 = devices::get_type<uint8_t>();
    uint16 = devices::get_type<uint16_t>();
    uint32 = devices::get_type<uint32_t>();
    uint64 = devices::get_type<uint64_t>();

    float32 = devices::get_type<float>();
    float64 = devices::get_type<double>();

    rational = implementations::ArbitraryPrecisionRationalType::get();

    // polynomial = implementations::PolyRationalArbitraryPrecisionType::get();
}

TypePtr BuiltinTypes::get_int(dimn_t bits) const
{
    switch (bits) {
        case 8: return int8;
        case 16: return int16;
        case 32: return int32;
        case 64: return int64;
        default:
            RPY_THROW(
                    std::invalid_argument,
                    string_cat(
                            "no builtin type for integer with ",
                            std::to_string(bits),
                            " bits"
                    )
            );
    }
}

TypePtr BuiltinTypes::get_uint(dimn_t bits) const
{
    switch (bits) {
        case 8: return uint8;
        case 16: return uint16;
        case 32: return uint32;
        case 64: return uint64;
        default:
            RPY_THROW(
                    std::invalid_argument,
                    string_cat(
                            "no builtin type for unsigned integer with ",
                            std::to_string(bits),
                            " bits"
                    )
            );
    }
}

TypePtr BuiltinTypes::get_rational(dimn_t bits) const
{
    (void) this;
    RPY_THROW(
            std::invalid_argument,
            string_cat(
                    "no builtin type for rational with ",
                    std::to_string(bits),
                    " bits"
            )
    );
    RPY_UNREACHABLE_RETURN(nullptr);
}

const BuiltinTypes rpy::scalars::builtin_types{};
