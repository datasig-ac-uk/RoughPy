//
// Created by sam on 16/04/24.
//

#ifndef UMINUS_KERNEL_H
#define UMINUS_KERNEL_H

#include "kernel.h"
#include "generic_kernel.h"
#include "argument_specs.h"

namespace rpy {
namespace algebra {

class UminusKernel;

extern template class VectorKernelBase<UminusKernel, dtl::MutableVectorArg, dtl::ConstVectorArg>;

class UminusKernel : public VectorKernelBase<UminusKernel, dtl::MutableVectorArg, dtl::ConstVectorArg> {
    using base_t = VectorKernelBase<UminusKernel, dtl::MutableVectorArg, dtl::ConstVectorArg>;

    public:

    RPY_NO_DISCARD
    string_view kernel_name() const noexcept;

    RPY_NO_DISCARD
    dtl::GenericUnaryFunction generic_op() const noexcept;

};

} // algebra
} // rpy

#endif //UMINUS_KERNEL_H
