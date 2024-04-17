//
// Created by sam on 16/04/24.
//

#ifndef UMINUS_KERNEL_H
#define UMINUS_KERNEL_H

#include "argument_specs.h"
#include "generic_kernel.h"
#include "kernel.h"

namespace rpy {
namespace algebra {

class UminusKernel;

extern template class VectorKernelBase<
        UminusKernel,
        dtl::MutableVectorArg,
        dtl::ConstVectorArg>;

class UminusKernel : public VectorKernelBase<
                             UminusKernel,
                             dtl::MutableVectorArg,
                             dtl::ConstVectorArg>
{

public:
    using VectorKernelBase::VectorKernelBase;

    RPY_NO_DISCARD string_view kernel_name() const noexcept;

    RPY_NO_DISCARD dtl::GenericUnaryFunction generic_op() const noexcept;
};

}// namespace algebra
}// namespace rpy

#endif// UMINUS_KERNEL_H
