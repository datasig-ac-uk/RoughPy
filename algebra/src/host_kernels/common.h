//
// Created by sam on 5/15/24.
//

#ifndef RPY_HOST_KERNELS_COMMON_H
#define RPY_HOST_KERNELS_COMMON_H

#include <roughpy/core/strings.h>
#include <roughpy/core/types.h>

#include <roughpy/device_support/host_kernel.h>
#include <roughpy/devices/kernel.h>

namespace rpy {
namespace algebra {
namespace dtl {

template <typename T>
struct IdOfTypeImpl;

template <>
struct IdOfTypeImpl<float> {
    static constexpr string_view value = "f32";
};

template <>
struct IdOfTypeImpl<double> {
    static constexpr string_view value = "f64";
};

template <typename T>
constexpr string_view IdOfType = IdOfTypeImpl<T>::value;

template <typename Op>
constexpr string_view NameOfOperator = Op::name;

template <typename T, typename... NameArgs>
devices::Kernel make_kernel(NameArgs&&... args)
{
    return devices::Kernel(new devices::HostKernel<T>(
            string_cat(std::forward<NameArgs>(args)...)
    ));
}

}// namespace dtl
}// namespace algebra
}// namespace rpy

#endif// RPY_HOST_KERNELS_COMMON_H
