//
// Created by sam on 4/16/24.
//

#include "fused_add_right_scalar_multiply.h"
#include "generic_kernel_MV_CV_CS.h"

namespace rpy {
namespace algebra {

template class VectorKernelBase<
        FusedAddRightScalarMultiply,
        dtl::MutableVectorArg,
        dtl::ConstVectorArg,
        dtl::ConstScalarArg>;

}// namespace algebra
}// namespace rpy

std::string_view
rpy::algebra::FusedAddRightScalarMultiply::kernel_name() const noexcept
{
    return "fused_add_scalar_right_mul";
}
