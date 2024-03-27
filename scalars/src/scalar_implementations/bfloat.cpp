//
// Created by sam on 3/26/24.
//

#include "bfloat.h"

#include "scalar_serialization.h"

#define RPY_SERIAL_IMPL_CLASSNAME rpy::scalars::BFloat16
#define RPY_SERIAL_EXTERNAL cereal
#define RPY_SERIAL_DO_SPLIT
#define RPY_SERIAL_NO_VERSION
#include <roughpy/platform/serialization_instantiations.inl>
