//
// Created by user on 06/11/23.
//

#include "half_type.h"

using namespace rpy;
using namespace rpy::scalars;

static constexpr RingCharacteristics
        half_ring_characteristics{true, true, true, false};

HalfType::HalfType() : base_t("half", "f16", half_ring_characteristics) {}
