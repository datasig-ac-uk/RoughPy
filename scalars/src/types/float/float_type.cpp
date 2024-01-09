//
// Created by user on 06/11/23.

#include "float_type.h"
#include "scalars_fwd.h"

#include <roughpy/platform/devices/types.h>

using namespace rpy;
using namespace rpy::scalars;

static constexpr RingCharacteristics
        float_ring_characteristics{true, true, true, false};

FloatType::FloatType() : base_t("SPReal", "f32", float_ring_characteristics) {}

const ScalarType* FloatType::get() noexcept
{
    static const FloatType type;
    return &type;
}

template <>
ROUGHPY_SCALARS_EXPORT optional<const ScalarType*>
scalars::dtl::ScalarTypeOfImpl<float>::get() noexcept
{
    return FloatType::get();
}
