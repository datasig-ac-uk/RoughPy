//
// Created by sam on 24/06/24.
//

#ifndef ROUGHPY_SCALAR_IMPLEMENTATIONS_HALF_H
#define ROUGHPY_SCALAR_IMPLEMENTATIONS_HALF_H


#include <Eigen/Core>

namespace rpy { namespace scalars { namespace implementations {

using Half = Eigen::half;



}}


namespace devices {
 namespace dtl {


template <>
struct type_code_of_impl<scalars::implementations::Half>
{
    static constexpr TypeCode value = TypeCode::Float;
};

template <>
struct type_id_of_impl<scalars::implementations::Half>
{
    static constexpr string_view value = "f16";
};

 }}

}


#endif //ROUGHPY_SCALAR_IMPLEMENTATIONS_HALF_H
