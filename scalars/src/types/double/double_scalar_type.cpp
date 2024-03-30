//
// Created by user on 06/11/23.
//

#include "double_scalar_type.h"

#include "double_type.h"

using namespace rpy;
using namespace rpy::scalars;

static constexpr RingCharacteristics
        double_ring_characteristics{true, true, true, false};

DoubleType::DoubleType()
    : ScalarType(
              &devices::double_type,
              devices::get_host_device(),
              double_ring_characteristics
      )
{}

// template <>
// ROUGHPY_SCALARS_EXPORT optional<const ScalarType*>
// scalars::dtl::ScalarTypeOfImpl<double>::get() noexcept
// {
//     return DoubleType::get();
// }

const DoubleType scalars::double_type;

const ScalarType* DoubleType::get() noexcept
{
    return &double_type;
    ;
}
