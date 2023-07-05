//
// Created by user on 26/04/23.
//

#include "b_float_16_type.h"

#include <string>
#include <utility>

using namespace rpy;
using namespace rpy::scalars;

BFloat16Type::BFloat16Type()
    : StandardScalarType<bfloat16>(
            string("BFloat16"), string("bf16"), sizeof(bfloat16),
            alignof(bfloat16),
            {ScalarTypeCode::BFloat, sizeof(bfloat16) * CHAR_BIT, 1U},
            {ScalarDeviceType::CPU, 0})
{}
