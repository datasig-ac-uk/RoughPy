//
// Created by user on 06/11/23.
//

#include "b_float_16_type.h"


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
