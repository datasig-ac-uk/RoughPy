//
// Created by user on 06/11/23.
//

#include "half_scalar_type.h"

#include "half_type.h"

using namespace rpy;
using namespace rpy::scalars;

static constexpr RingCharacteristics
        half_ring_characteristics{true, true, true, false};

HalfType::HalfType()
    : ScalarType(
              &devices::half_type,
              devices::get_host_device(),
              half_ring_characteristics
      )
{}

const HalfType scalars::half_type;

const ScalarType* HalfType::get() noexcept { return &half_type; }

// template <>
// ROUGHPY_SCALARS_EXPORT optional<const ScalarType*>
// scalars::dtl::ScalarTypeOfImpl<devices::half>::get() noexcept
// {
//     return HalfType::get();
// }
