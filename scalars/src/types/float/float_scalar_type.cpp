//
// Created by user on 06/11/23.

#include "float_scalar_type.h"
#include "devices/fundamental_types/float_type.h"

using namespace rpy;
using namespace rpy::scalars;

static constexpr RingCharacteristics
        float_ring_characteristics{true, true, true, false};

FloatType::FloatType()
    : ScalarType(
              &devices::float_type,
              devices::get_host_device(),
              float_ring_characteristics
      )
{}

const FloatType scalars::float_type;

const ScalarType* FloatType::get() noexcept { return &float_type; }

// template <>
// ROUGHPY_SCALARS_EXPORT optional<const ScalarType*>
// scalars::dtl::ScalarTypeOfImpl<float>::get() noexcept
// {
//     return FloatType::get();
// }
