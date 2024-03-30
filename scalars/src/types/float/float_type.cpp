//
// Created by sam on 3/30/24.
//

#include "float_type.h"


using namespace rpy;
using namespace rpy::devices;

FloatType::FloatType() : Type("f32", "SPReal", type_info<float>())
{}

const FloatType devices::float_type;
