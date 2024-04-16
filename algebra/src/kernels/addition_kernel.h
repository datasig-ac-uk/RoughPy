//
// Created by sam on 4/16/24.
//

#ifndef ADDITION_KERNEL_H
#define ADDITION_KERNEL_H

#include "arg_data.h"
#include "argument_specs.h"
#include "generic_kernel.h"
#include "kernel.h"

namespace rpy {
namespace algebra {

class AdditionKernel;

extern template class VectorKernelBase<
        AdditionKernel,
        dtl::MutableVectorArg,
        dtl::ConstVectorArg,
        dtl::ConstVectorArg>;

class AdditionKernel : public VectorKernelBase<
                               AdditionKernel,
                               dtl::MutableVectorArg,
                               dtl::ConstVectorArg,
                               dtl::ConstVectorArg>
{
public:

    using VectorKernelBase::VectorKernelBase;

    string_view kernel_name() const noexcept;

    dtl::GenericBinaryFunction generic_op() const noexcept
    {
        return [](scalars::Scalar& out, const scalars::Scalar& left, const scalars::Scalar& right) {
            out = left + right;
        };
    }
};

}// namespace algebra
}// namespace rpy

#endif// ADDITION_KERNEL_H
