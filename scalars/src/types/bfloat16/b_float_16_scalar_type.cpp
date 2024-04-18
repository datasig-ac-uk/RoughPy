//
// Created by user on 06/11/23.
//

#include "b_float_16_scalar_type.h"

#include "devices/fundamental_types/b_float_16_type.h"
#include "scalar_implementations/bfloat.h"

using namespace rpy;
using namespace rpy::scalars;

static constexpr RingCharacteristics
        bf16_ring_characteristics{true, true, true, false};

BFloat16ScalarType::BFloat16ScalarType()
    : ScalarType(
              devices::BFloat16Type::get(),
              devices::get_host_device(),
              bf16_ring_characteristics
      )
{}


const ScalarType* BFloat16ScalarType::get() noexcept
{
    static const BFloat16ScalarType type;
    return &type;
}

// template <>
// ROUGHPY_SCALARS_EXPORT optional<const ScalarType*>
// scalars::dtl::ScalarTypeOfImpl<devices::bfloat16>::get() noexcept
// {
//     return BFloat16Type::get();
// }
