//
// Created by sam on 4/16/24.
//

#include "addition_kernel.h"
#include "generic_kernel_MV_CV_CV.h"

namespace rpy {
namespace algebra {

template class VectorKernelBase<
        AdditionKernel,
        dtl::MutableVectorArg,
        dtl::ConstVectorArg,
        dtl::ConstVectorArg>;

}// namespace algebra
}// namespace rpy

std::string_view rpy::algebra::AdditionKernel::kernel_name() const noexcept
{
    return "addition";
}
