//
// Created by sam on 16/04/24.
//

#include "inplace_right_scalar_multiply_kernel.h"
#include "generic_kernel_MV_CS.h"

namespace rpy {
namespace algebra {

template class VectorKernelBase<
        InplaceRightScalarMultiplyKernel,
        dtl::MutableVectorArg,
        dtl::ConstScalarArg>;

}// namespace algebra
}// namespace rpy

std::string_view
rpy::algebra::InplaceRightScalarMultiplyKernel::kernel_name() const noexcept
{
    return "inplace_right_scalar_multiply";
}
