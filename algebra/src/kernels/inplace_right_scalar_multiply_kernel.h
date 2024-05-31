//
// Created by sam on 16/04/24.
//

#ifndef INPLACE_RIGHT_SCALAR_MULTIPLY_KERNEL_H
#define INPLACE_RIGHT_SCALAR_MULTIPLY_KERNEL_H
#include "argument_specs.h"
#include "common.h"
#include "generic_kernel.h"
#include "kernel.h"

namespace rpy {
namespace algebra {

class InplaceRightScalarMultiplyKernel;
extern template class VectorKernelBase<
        InplaceRightScalarMultiplyKernel,
        dtl::MutableVectorArg,
        dtl::ConstScalarArg>;

class InplaceRightScalarMultiplyKernel
    : public VectorKernelBase<
              InplaceRightScalarMultiplyKernel,
              dtl::MutableVectorArg,
              dtl::ConstScalarArg>
{
    using base_t = VectorKernelBase<
            InplaceRightScalarMultiplyKernel,
            dtl::MutableVectorArg,
            dtl::ConstScalarArg>;

public:
    using base_t::base_t;

    RPY_NO_DISCARD string_view kernel_name() const noexcept;
    RPY_NO_DISCARD dtl::GenericUnaryFunction generic_op() const noexcept
    {
        return [](scalars::ScalarRef out, scalars::ScalarCRef multiplier) {
            out *= multiplier;
        };
    }
};

}// namespace algebra
}// namespace rpy

#endif// INPLACE_RIGHT_SCALAR_MULTIPLY_KERNEL_H
