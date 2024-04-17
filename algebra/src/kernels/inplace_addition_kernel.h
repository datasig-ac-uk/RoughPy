//
// Created by sam on 16/04/24.
//

#ifndef INPLACE_ADDITION_KERNEL_H
#define INPLACE_ADDITION_KERNEL_H

#include "arg_data.h"
#include "argument_specs.h"
#include "generic_kernel.h"
#include "kernel.h"

namespace rpy {
namespace algebra {

class InplaceAdditionKernel;

extern template class VectorKernelBase<
        InplaceAdditionKernel,
        dtl::MutableVectorArg,
        dtl::ConstVectorArg>;

class InplaceAdditionKernel : public VectorKernelBase<
                                      InplaceAdditionKernel,
                                      dtl::MutableVectorArg,
                                      dtl::ConstVectorArg>
{
    static constexpr string_view s_kernel_name = "inplace_add";

    using base_t = VectorKernelBase<
            InplaceAdditionKernel,
            dtl::MutableVectorArg,
            dtl::ConstVectorArg>;

public:
    using base_t::base_t;

    RPY_NO_DISCARD string_view kernel_name() const noexcept
    {
        return s_kernel_name;
    }

    RPY_NO_DISCARD dtl::GenericUnaryFunction generic_op() const noexcept
    {
        return [](scalars::Scalar& out, const scalars::Scalar& arg) {
            out += arg;
        };
    }
};

}// namespace algebra
}// namespace rpy

#endif// INPLACE_ADDITION_KERNEL_H
