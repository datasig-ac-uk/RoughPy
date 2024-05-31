//
// Created by sam on 4/17/24.
//

#ifndef FUSED_SUB_RIGHT_SCALAR_MULTIPLY_KERNEL_H
#define FUSED_SUB_RIGHT_SCALAR_MULTIPLY_KERNEL_H

#include "argument_specs.h"
#include "generic_kernel.h"
#include "kernel.h"

namespace rpy {
namespace algebra {

class FusedSubRightScalarMultiplyKernel;

extern template class VectorKernelBase<
        FusedSubRightScalarMultiplyKernel,
        dtl::MutableVectorArg,
        dtl::ConstVectorArg,
        dtl::ConstScalarArg>;

class FusedSubRightScalarMultiplyKernel
    : public VectorKernelBase<
              FusedSubRightScalarMultiplyKernel,
              dtl::MutableVectorArg,
              dtl::ConstVectorArg,
              dtl::ConstScalarArg>
{
public:
    using VectorKernelBase::VectorKernelBase;

    string_view kernel_name() const noexcept;

    auto generic_op() const noexcept
    {
        return [](scalars::ScalarRef out,
                  scalars::ScalarCRef arg,
                  scalars::ScalarCRef multiplier) {
            out -= (arg * multiplier);
        };
    }
};

}// namespace algebra
}// namespace rpy

#endif// FUSED_SUB_RIGHT_SCALAR_MULTIPLY_KERNEL_H
