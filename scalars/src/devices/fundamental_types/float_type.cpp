//
// Created by sam on 3/30/24.
//

#include "float_type.h"

using namespace rpy;
using namespace rpy::devices;

FloatType::FloatType() : FundamentalType("f32", "SPReal") {}

const FloatType* FloatType::get() noexcept
{
    static const FloatType type;
    return &type;
}
