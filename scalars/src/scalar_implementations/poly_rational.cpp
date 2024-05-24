//
// Created by sam on 3/25/24.
//

#include "poly_rational.h"
#include "arbitrary_precision_rational.h"
// #include "scalar_serialization.h"

using namespace rpy;
using namespace rpy::scalars;

namespace rpy {
namespace scalars {

template class Polynomial<ArbitraryPrecisionRational>;
template class Polynomial<Rational64>;
template class Polynomial<Rational32>;

}// namespace scalars
}// namespace rpy

// #define RPY_SERIAL_IMPL_CLASSNAME rpy::scalars::PolyRat32
// #define RPY_SERIAL_DO_SPLIT
// #include <roughpy/platform/serialization_instantiations.inl>
//
// #define RPY_SERIAL_IMPL_CLASSNAME rpy::scalars::PolyRat64
// #define RPY_SERIAL_DO_SPLIT
// #include <roughpy/platform/serialization_instantiations.inl>
//
// #define RPY_SERIAL_IMPL_CLASSNAME rpy::scalars::APPolyRat
// #define RPY_SERIAL_DO_SPLIT
// #include <roughpy/platform/serialization_instantiations.inl>
