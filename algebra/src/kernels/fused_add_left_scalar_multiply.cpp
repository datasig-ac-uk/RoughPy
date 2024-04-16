//
// Created by sam on 4/16/24.
//

#include "fused_add_left_scalar_multiply.h"
#include "generic_kernel_MV_CV_CS.h"

namespace rpy {
namespace algebra {

template class VectorKernelBase<
        FusedAddLeftScalarMultiply,
        dtl::MutableVectorArg,
        dtl::ConstVectorArg,
        dtl::ConstScalarArg>;

}// namespace algebra
}// namespace rpy

std::string_view
rpy::algebra::FusedAddLeftScalarMultiply::kernel_name() const noexcept
{
    return "fused_add_scalar_left_mul";
}
