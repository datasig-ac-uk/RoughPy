//
// Created by user on 06/11/23.
//

#include "b_float_16_type.h"
#include "scalar_implementations/bfloat.h"

using namespace rpy;
using namespace rpy::scalars;


static constexpr RingCharacteristics bf16_ring_characteristics {
    true,
    true,
     true,
    false
};


BFloat16Type::BFloat16Type()
    : base_t("bfloat16", "bf16", bf16_ring_characteristics)
{}



const BFloat16Type scalars::bfloat16_type;

const ScalarType* BFloat16Type::get() noexcept
{
    return &bfloat16_type;
}

// template <>
// ROUGHPY_SCALARS_EXPORT optional<const ScalarType*>
// scalars::dtl::ScalarTypeOfImpl<devices::bfloat16>::get() noexcept
// {
//     return BFloat16Type::get();
// }
