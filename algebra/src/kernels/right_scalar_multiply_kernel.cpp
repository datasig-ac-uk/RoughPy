//
// Created by sam on 16/04/24.
//

#include "right_scalar_multiply_kernel.h"

namespace rpy {
namespace algebra {

template class VectorKernelBase<
        RightScalarMultiplyKernel,
        dtl::MutableVectorArg,
        dtl::ConstVectorArg,
        dtl::ConstScalarArg>;

}// namespace algebra
}// namespace rpy

std::string_view
rpy::algebra::RightScalarMultiplyKernel::kernel_name() const noexcept
{
    return "right_scalar_multiply";
}
