//
// Created by user on 06/11/23.
//

#include "double_type.h"


using namespace rpy;
using namespace rpy::scalars;

static constexpr RingCharacteristics
        double_ring_characteristics{true, true, true, false};

DoubleType::DoubleType()
    : base_t("double", "f64", double_ring_characteristics)
{}
