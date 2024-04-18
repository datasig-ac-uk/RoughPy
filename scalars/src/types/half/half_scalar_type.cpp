//
// Created by user on 06/11/23.
//

#include "half_scalar_type.h"

#include "devices/fundamental_types/half_type.h"

using namespace rpy;
using namespace rpy::scalars;

static constexpr RingCharacteristics
        half_ring_characteristics{true, true, true, false};

HalfType::HalfType()
    : ScalarType(
              devices::HalfType::get(),
              devices::get_host_device(),
              half_ring_characteristics
      )
{}

const ScalarType* HalfType::get() noexcept
{
    static const HalfType type;
    return &type;
}

// template <>
// ROUGHPY_SCALARS_EXPORT optional<const ScalarType*>
// scalars::dtl::ScalarTypeOfImpl<devices::half>::get() noexcept
// {
//     return HalfType::get();
// }
