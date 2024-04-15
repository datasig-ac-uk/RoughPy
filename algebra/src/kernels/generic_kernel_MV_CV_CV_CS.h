//
// Created by sam on 15/04/24.
//

#ifndef GENERIC_KERNEL_VS_H
#define GENERIC_KERNEL_VS_H

#include "arg_data.h"
#include "argument_specs.h"
#include "common.h"
#include "generic_kernel.h"

namespace rpy {
namespace algebra {
namespace dtl {

template <>
class GenericKernel<
        MutableVectorArg,
        ConstVectorArg,
        ConstVectorArg,
        ConstScalarArg>
{

};

}// namespace dtl
}// namespace algebra
}// namespace rpy

#endif// GENERIC_KERNEL_VS_H
