//
// Created by sam on 16/04/24.
//

#include "uminus_kernel.h"
#include "generic_kernel_MV_CV.h"

namespace rpy {
namespace algebra {


template class VectorKernelBase<UminusKernel, dtl::MutableVectorArg, dtl::ConstVectorArg>;

} // algebra
}// namespace rpy

std::string_view rpy::algebra::UminusKernel::kernel_name() const noexcept
{
    return "uminus";
}
rpy::algebra::dtl::GenericUnaryFunction
rpy::algebra::UminusKernel::generic_op() const noexcept
{
    return [](scalars::Scalar& out, const scalars::Scalar& arg) {
        out = -arg;
    };
}
