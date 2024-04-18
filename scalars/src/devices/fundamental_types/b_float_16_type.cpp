//
// Created by sam on 3/30/24.
//

#include "b_float_16_type.h"

using namespace rpy;
using namespace rpy::devices;

BFloat16Type::BFloat16Type() : FundamentalType("bf16", "BFloat16") {}

const BFloat16Type* BFloat16Type::get() noexcept
{
    static const BFloat16Type type;
    return &type;
}
