//
// Created by sam on 16/04/24.
//

#ifndef INPLACE_SUBTRACTION_KERNEL_H
#define INPLACE_SUBTRACTION_KERNEL_H

#include "arg_data.h"
#include "argument_specs.h"
#include "generic_kernel.h"
#include "kernel.h"

#include <functional>

namespace rpy {
namespace algebra {

class InplaceSubtractionKernel;

extern template class VectorKernelBase<
        InplaceSubtractionKernel,
        dtl::MutableVectorArg,
        dtl::ConstVectorArg>;

class InplaceSubtractionKernel : public VectorKernelBase<
                                         InplaceSubtractionKernel,
                                         dtl::MutableVectorArg,
                                         dtl::ConstVectorArg>
{
    static constexpr string_view s_kernel_name = "inplace_sub";

    using base_t = VectorKernelBase<
            InplaceSubtractionKernel,
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
        return [](scalars::ScalarRef out, scalars::ScalarCRef arg) {
            out -= arg;
        };
    }
};

}// namespace algebra
}// namespace rpy

#endif// INPLACE_SUBTRACTION_KERNEL_H
