//
// Created by sam on 4/16/24.
//

#ifndef SUBTRACTION_KERNEL_H
#define SUBTRACTION_KERNEL_H

#include "arg_data.h"
#include "argument_specs.h"
#include "generic_kernel.h"
#include "kernel.h"

namespace rpy {
namespace algebra {

class SubtractionKernel;

extern template class VectorKernelBase<
        SubtractionKernel,
        dtl::MutableVectorArg,
        dtl::ConstVectorArg,
        dtl::ConstVectorArg>;

class SubtractionKernel : public VectorKernelBase<
                                  SubtractionKernel,
                                  dtl::MutableVectorArg,
                                  dtl::ConstVectorArg,
                                  dtl::ConstVectorArg>
{
public:
    using VectorKernelBase::VectorKernelBase;

    string_view kernel_name() const noexcept;

    dtl::GenericBinaryFunction generic_op() const noexcept
    {
        return [](scalars::Scalar& out,
                  const scalars::Scalar& left,
                  const scalars::Scalar& right) { out = left - right; };
    }
};

}// namespace algebra
}// namespace rpy

#endif// SUBTRACTION_KERNEL_H
