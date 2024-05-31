//
// Created by sam on 16/04/24.
//

#ifndef RIGHT_SCALAR_MULTIPLY_KERNEL_H
#define RIGHT_SCALAR_MULTIPLY_KERNEL_H

#include "argument_specs.h"
#include "common.h"
#include "generic_kernel.h"
#include "kernel.h"

namespace rpy {
namespace algebra {

class RightScalarMultiplyKernel;
extern template class VectorKernelBase<
        RightScalarMultiplyKernel,
        dtl::MutableVectorArg,
        dtl::ConstVectorArg,
        dtl::ConstScalarArg>;

class RightScalarMultiplyKernel : public VectorKernelBase<
                                         RightScalarMultiplyKernel,
                                         dtl::MutableVectorArg,
                                         dtl::ConstVectorArg,
                                         dtl::ConstScalarArg>
{
    using base_t = VectorKernelBase<
            RightScalarMultiplyKernel,
            dtl::MutableVectorArg,
            dtl::ConstVectorArg,
            dtl::ConstScalarArg>;

public:
    using base_t::base_t;

    RPY_NO_DISCARD string_view kernel_name() const noexcept;
    RPY_NO_DISCARD dtl::GenericBinaryFunction generic_op() const noexcept
    {
        return [](scalars::ScalarRef out,
                  const scalars::ScalarCRef arg,
                  const scalars::ScalarCRef multiplier) {
            out = arg * multiplier;
        };
    }
};

}// namespace algebra
}// namespace rpy

#endif// RIGHT_SCALAR_MULTIPLY_KERNEL_H
