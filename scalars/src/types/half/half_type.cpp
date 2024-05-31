//
// Created by sam on 3/30/24.
//

#include "half_type.h"

using namespace rpy;
using namespace rpy::devices;

HalfType::HalfType()
    : Type("f16",
           "HPReal",
           {TypeCode::Float, sizeof(scalars::Half), alignof(scalars::Half), 1},
           traits_of<scalars::Half>())
{}

const HalfType* HalfType::get() noexcept
{
    static const HalfType type;
    return &type;
}
