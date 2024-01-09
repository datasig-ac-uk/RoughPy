//
// Created by user on 06/11/23.
//

#include "half_type.h"

using namespace rpy;
using namespace rpy::scalars;

static constexpr RingCharacteristics
        half_ring_characteristics{true, true, true, false};

HalfType::HalfType() : base_t("HPReal", "f16", half_ring_characteristics) {}

const ScalarType* HalfType::get() noexcept
{
    static const HalfType type;
    return &type;
}

template <>
ROUGHPY_SCALARS_EXPORT optional<const ScalarType*>
scalars::dtl::ScalarTypeOfImpl<devices::half>::get() noexcept
{
    return HalfType::get();
}
