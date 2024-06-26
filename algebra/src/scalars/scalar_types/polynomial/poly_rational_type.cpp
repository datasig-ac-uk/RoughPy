//
// Created by sam on 24/06/24.
//

#include "poly_rational_type.h"

namespace rpy {
namespace scalars {
namespace implementations {

template class PolyRationalType<Rational32>;
template class PolyRationalType<Rational64>;
template class PolyRationalType<ArbitraryPrecisionRational>;

} // implementations
} // scalars
} // rpy