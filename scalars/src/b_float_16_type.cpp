//
// Created by user on 26/04/23.
//

#include "b_float_16_type.h"

using namespace rpy;
using namespace rpy::scalars;

BFloat16Type::BFloat16Type()
    : StandardScalarType<bfloat16>("bf16", "bfloat16")
{
}
