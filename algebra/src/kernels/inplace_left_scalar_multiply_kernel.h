//
// Created by sam on 16/04/24.
//

#ifndef INPLACE_LEFT_SCALAR_MULTIPLY_KERNEL_H
#define INPLACE_LEFT_SCALAR_MULTIPLY_KERNEL_H

#include "argument_specs.h"
#include "common.h"
#include "generic_kernel.h"
#include "kernel.h"

namespace rpy {
namespace algebra {

class InplaceLeftScalarMultiplyKernel;
extern template class VectorKernelBase<
        InplaceLeftScalarMultiplyKernel,
        dtl::MutableVectorArg,
        dtl::ConstScalarArg>;

class InplaceLeftScalarMultiplyKernel : public VectorKernelBase<
                                         InplaceLeftScalarMultiplyKernel,
                                         dtl::MutableVectorArg,
                                         dtl::ConstScalarArg>
{
    using base_t = VectorKernelBase<
            InplaceLeftScalarMultiplyKernel,
            dtl::MutableVectorArg,
            dtl::ConstScalarArg>;

public:
    using base_t::base_t;

    RPY_NO_DISCARD string_view kernel_name() const noexcept;
    RPY_NO_DISCARD dtl::GenericUnaryFunction generic_op() const noexcept
    {
        return [](scalars::Scalar& out,
                  const scalars::Scalar& multiplier) {
            out = multiplier*out;
        };
    }
};

}// namespace algebra
}// namespace rpy

#endif// INPLACE_LEFT_SCALAR_MULTIPLY_KERNEL_H
