//
// Created by sam on 16/04/24.
//

#include "left_scalar_multiply_kernel.h"
#include "generic_kernel_MV_CV_CS.h"

namespace rpy {
namespace algebra {

template class VectorKernelBase<
        LeftScalarMultiplyKernel,
        dtl::MutableVectorArg,
        dtl::ConstVectorArg,
        dtl::ConstScalarArg>;

}// namespace algebra
}// namespace rpy

std::string_view
rpy::algebra::LeftScalarMultiplyKernel::kernel_name() const noexcept
{
    return "left_scalar_multiply";
}
