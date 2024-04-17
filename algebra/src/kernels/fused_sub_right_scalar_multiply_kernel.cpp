//
// Created by sam on 4/17/24.
//

#include "fused_sub_right_scalar_multiply_kernel.h"
#include "generic_kernel_MV_CV_CS.h"

namespace rpy {
namespace algebra {

template class VectorKernelBase<
        FusedSubRightScalarMultiplyKernel,
        dtl::MutableVectorArg,
        dtl::ConstVectorArg,
        dtl::ConstScalarArg>;

}// namespace algebra
}// namespace rpy

std::string_view
rpy::algebra::FusedSubRightScalarMultiplyKernel::kernel_name() const noexcept
{
    return "fused_sub_scalar_right_mul";
}
