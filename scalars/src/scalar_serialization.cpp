//
// Created by user on 02/10/23.
//

#include <roughpy/scalars/scalars_fwd.h>

using namespace rpy;
using namespace rpy::scalars;


#define RPY_SERIAL_IMPL_CLASSNAME rpy::scalars::half
#define RPY_SERIAL_EXTERNAL cereal
#define RPY_SERIAL_DO_SPLIT
#include <roughpy/platform/serialization_instantiations.inl>

#define RPY_SERIAL_IMPL_CLASSNAME rpy::scalars::bfloat16
#define RPY_SERIAL_EXTERNAL cereal
#define RPY_SERIAL_DO_SPLIT
#include <roughpy/platform/serialization_instantiations.inl>

#define RPY_SERIAL_IMPL_CLASSNAME rpy::scalars::indeterminate_type
#define RPY_SERIAL_EXTERNAL cereal
#define RPY_SERIAL_DO_SPLIT
#include <roughpy/platform/serialization_instantiations.inl>

#define RPY_SERIAL_IMPL_CLASSNAME rpy::scalars::monomial
#define RPY_SERIAL_EXTERNAL cereal
#define RPY_SERIAL_DO_SPLIT
#include <roughpy/platform/serialization_instantiations.inl>

#define RPY_SERIAL_IMPL_CLASSNAME rpy::scalars::rational_poly_scalar
#define RPY_SERIAL_EXTERNAL cereal
#define RPY_SERIAL_DO_SPLIT
#include <roughpy/platform/serialization_instantiations.inl>