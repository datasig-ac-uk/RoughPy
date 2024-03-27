//
// Created by sam on 3/26/24.
//

#ifndef RATIONAL_H
#define RATIONAL_H

#include <boost/rational.hpp>

namespace boost {

extern template class ROUGHPY_SCALARS_NO_EXPORT rational<int64_t>;
extern template class ROUGHPY_SCALARS_NO_EXPORT rational<int32_t>;

}// namespace boost

namespace rpy {
namespace scalars {

template <typename I>
using Rational = boost::rational<I>;

using Rational64 = boost::rational<int64_t>;
using Rational32 = boost::rational<int32_t>;

}// namespace scalars


namespace devices {
namespace dtl {

template <typename T>
struct type_code_of_impl<scalars::Rational<T>>
{
    static constexpr TypeCode value = TypeCode::Rational;
};

template <typename T>
struct type_size_of_impl<scalars::Rational<T>>
{
    static constexpr dimn_t value = sizeof(T);
};


}
}


}// namespace rpy

#endif// RATIONAL_H
