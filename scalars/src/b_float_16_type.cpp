//
// Created by user on 26/04/23.
//

#include "b_float_16_type.h"

using namespace rpy;
using namespace rpy::scalars;

static const ScalarTypeInfo bfloat16_type_info {
        "bfloat16",
        "bf16",
        sizeof(bfloat16),
        alignof(bfloat16),
        {
                ScalarTypeCode::BFloat,
                sizeof(bfloat16)*CHAR_BIT,
                1
        },
        {
                ScalarDeviceType::CPU,
                0
        }
};


BFloat16Type::BFloat16Type() : StandardScalarType<bfloat16>(bfloat16_type_info)
{}
