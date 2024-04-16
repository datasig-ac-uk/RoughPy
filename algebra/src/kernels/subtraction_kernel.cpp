//
// Created by sam on 4/16/24.
//

#include "subtraction_kernel.h"

namespace rpy {
namespace algebra {

template class VectorKernelBase<
        SubtractionKernel,
        dtl::MutableVectorArg,
        dtl::ConstVectorArg,
        dtl::ConstVectorArg>;

}// namespace algebra
}// namespace rpy

std::string_view rpy::algebra::SubtractionKernel::kernel_name() const noexcept
{
    return "subtraction";
}
