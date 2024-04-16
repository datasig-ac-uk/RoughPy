//
// Created by sam on 16/04/24.
//

#include "inplace_left_scalar_multiply_kernel.h"
#include "generic_kernel_MV_CS.h"

namespace rpy {
namespace algebra {

template class VectorKernelBase<
        InplaceLeftScalarMultiplyKernel,
        dtl::MutableVectorArg,
        dtl::ConstScalarArg>;

}// namespace algebra
}// namespace rpy

rpy::string_view
rpy::algebra::InplaceLeftScalarMultiplyKernel::kernel_name() const noexcept
{
    return "inplace_left_scalar_multiply";
}
