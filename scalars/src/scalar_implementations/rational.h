//
// Created by sam on 3/26/24.
//

#ifndef RATIONAL_H
#define RATIONAL_H

#include <boost/rational.hpp>

namespace rpy {
namespace scalars {

template <typename I>
using Rational = boost::rational<I>;

extern template class ROUGHPY_SCALARS_NO_EXPORT boost::rational<int64_t>;
extern template class ROUGHPY_SCALARS_NO_EXPORT boost::rational<int32_t>;

using Rational64 = boost::rational<int64_t>;
using Rational32 = boost::rational<int32_t>;

}// namespace scalars
}// namespace rpy

#endif// RATIONAL_H
