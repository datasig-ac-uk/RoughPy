//
// Created by sam on 3/26/24.
//

#ifndef BFLOAT_H
#define BFLOAT_H

#include <roughpy/devices/core.h>
#include <Eigen/Core>

namespace rpy {
namespace scalars {

using BFloat16 = Eigen::bfloat16;

}

namespace devices {
namespace dtl {

template <>
struct type_code_of_impl<scalars::BFloat16> {
    static constexpr TypeCode value = TypeCode::BFloat;
};

template <>
struct type_id_of_impl<scalars::BFloat16>
{
    static constexpr string_view value = "bf16";
};

}// namespace dtl
}// namespace devices

}// namespace rpy

#endif// BFLOAT_H
