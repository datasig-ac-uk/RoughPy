//
// Created by sam on 24/06/24.
//

#ifndef ROUGHPY_SCALAR_TYPES_RATIONAL_H
#define ROUGHPY_SCALAR_TYPES_RATIONAL_H

#include <boost/rational.hpp>

#include <roughpy/core/types.h>

namespace rpy {
namespace scalars {
namespace implementations {

template <typename I>
using Rational = boost::rational<I>;

using Rational32 = Rational<int32_t>;
using Rational64 = Rational<int64_t>;

}// namespace implementations
}// namespace scalars

namespace devices {
namespace dtl {

template <>
struct type_id_of_impl<scalars::implementations::Rational<int64_t>> {
    static constexpr string_view value = "r64";
};

template <>
struct type_id_of_impl<scalars::implementations::Rational<int32_t>> {
    static constexpr string_view value = "r32";
};

template <typename T>
struct type_code_of_impl<scalars::implementations::Rational<T>> {
    static constexpr TypeCode value = TypeCode::Rational;
};

template <typename T>
struct type_size_of_impl<scalars::implementations::Rational<T>> {
    static constexpr dimn_t value = sizeof(T);
};

}// namespace dtl
}// namespace devices

}// namespace rpy

#endif// ROUGHPY_SCALAR_TYPES_RATIONAL_H
