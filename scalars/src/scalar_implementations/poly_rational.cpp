//
// Created by sam on 3/25/24.
//

#include "poly_rational.h"

using namespace rpy;
using namespace rpy::scalars;




namespace rpy { namespace scalars {

template class Polynomial<ArbitraryPrecisionRational>;
template class Polynomial<Rational64>;
template class Polynomial<Rational32>;

}}
