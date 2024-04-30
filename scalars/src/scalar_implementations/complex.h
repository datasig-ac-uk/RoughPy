//
// Created by sam on 3/26/24.
//

#ifndef COMPLEX_H
#define COMPLEX_H

#include <roughpy/devices/core.h>
#include <complex>

namespace rpy {
namespace scalars {

using Complex32 = std::complex<float>;
using Complex64 = std::complex<double>;

}// namespace scalars

namespace devices {
namespace dtl {

template <>
struct type_code_of_impl<scalars::Complex32> {
    static constexpr TypeCode value = TypeCode::Complex;
};

template <>
struct type_code_of_impl<scalars::Complex64> {
    static constexpr TypeCode value = TypeCode::Complex;
};

template <typename T>
struct type_size_of_impl<std::complex<T>> {
    static constexpr dimn_t value = sizeof(T);
};

}// namespace dtl
}// namespace devices

}// namespace rpy

#endif// COMPLEX_H
