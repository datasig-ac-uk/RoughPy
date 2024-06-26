//
// Created by sam on 24/06/24.
//

#ifndef ROUGHPY_SCALAR_IMPLEMENTATIONS_BLFOAT16_H
#define ROUGHPY_SCALAR_IMPLEMENTATIONS_BLFOAT16_H


#include <roughpy/devices/core.h>
#include <roughpy/devices/type.h>


#include <Eigen/Core>

namespace rpy { namespace scalars { namespace implementations {

using BFloat16 = Eigen::bfloat16;

}}

namespace devices {
namespace dtl {

template <>
struct type_code_of_impl<scalars::implementations::BFloat16> {
    static constexpr TypeCode value = TypeCode::BFloat;
};

template <>
struct type_id_of_impl<scalars::implementations::BFloat16>
{
    static constexpr string_view value = "bf16";
};

}
}

}

#endif //ROUGHPY_SCALAR_IMPLEMENTATIONS_BLFOAT16_H
