//
// Created by sam on 16/04/24.
//

#ifndef LEFT_SCALAR_MULTIPLY_KERNEL_H
#define LEFT_SCALAR_MULTIPLY_KERNEL_H

#include "argument_specs.h"
#include "common.h"
#include "generic_kernel.h"
#include "kernel.h"

namespace rpy {
namespace algebra {

class LeftScalarMultiplyKernel;
extern template class VectorKernelBase<
        LeftScalarMultiplyKernel,
        dtl::MutableVectorArg,
        dtl::ConstVectorArg,
        dtl::ConstScalarArg>;

class LeftScalarMultiplyKernel : public VectorKernelBase<
                                         LeftScalarMultiplyKernel,
                                         dtl::MutableVectorArg,
                                         dtl::ConstVectorArg,
                                         dtl::ConstScalarArg>
{
    using base_t = VectorKernelBase<
            LeftScalarMultiplyKernel,
            dtl::MutableVectorArg,
            dtl::ConstVectorArg,
            dtl::ConstScalarArg>;

public:
    using base_t::base_t;

    RPY_NO_DISCARD string_view kernel_name() const noexcept;
    RPY_NO_DISCARD dtl::GenericBinaryFunction generic_op() const noexcept
    {
        return [](scalars::Scalar& out,
                  const scalars::Scalar& arg,
                  const scalars::Scalar& multiplier) {
            out = multiplier * arg;
        };
    }
};

}// namespace algebra
}// namespace rpy

#endif// LEFT_SCALAR_MULTIPLY_KERNEL_H
