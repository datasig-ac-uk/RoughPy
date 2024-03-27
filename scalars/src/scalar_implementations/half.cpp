//
// Created by sam on 3/26/24.
//

#include "half.h"
#include "scalar_serialization.h"

#define RPY_EXPORT_MACRO ROUGHPY_SCALARS_EXPORT
#define RPY_SERIAL_IMPL_CLASSNAME rpy::scalars::Half
#define RPY_SERIAL_EXTERNAL cereal
#define RPY_SERIAL_DO_SPLIT
#define RPY_SERIAL_NO_VERSION
#include <roughpy/platform/serialization_instantiations.inl>
