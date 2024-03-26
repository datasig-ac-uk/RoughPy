//
// Created by sam on 3/26/24.
//

#ifndef ARBITRARY_PRECISION_RATIONAL_H
#define ARBITRARY_PRECISION_RATIONAL_H

#include <boost/multiprecision/gmp.hpp>


namespace rpy { namespace scalars {

extern template class ROUGHPY_SCALARS_NO_EXPORT boost::multiprecision::mpq_rational;

using ArbitraryPrecisionRational = boost::multiprecision::mpq_rational;

}}

#endif //ARBITRARY_PRECISION_RATIONAL_H
