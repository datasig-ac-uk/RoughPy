//
// Created by user on 06/11/23.
//

#include "double_scalar_type.h"

#include "devices/fundamental_types/double_type.h"

#include <gtest/internal/gtest-internal.h>

using namespace rpy;
using namespace rpy::scalars;

static constexpr RingCharacteristics
        double_ring_characteristics{true, true, true, false};

DoubleType::DoubleType()
    : ScalarType(
              devices::DoubleType::get(),
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

const ScalarType* DoubleType::get() noexcept
{
    static const DoubleType type;
    return &type;
    ;
}
