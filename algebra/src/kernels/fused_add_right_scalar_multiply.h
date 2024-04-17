//
// Created by sam on 4/16/24.
//

#ifndef FUSED_ADD_RIGHT_SCALAR_MULTIPLY_H
#define FUSED_ADD_RIGHT_SCALAR_MULTIPLY_H

#include "argument_specs.h"
#include "common.h"
#include "generic_kernel.h"
#include "kernel.h"

namespace rpy {
namespace algebra {

class FusedAddRightScalarMultiply;

extern template class VectorKernelBase<
        FusedAddRightScalarMultiply,
        dtl::MutableVectorArg,
        dtl::ConstVectorArg,
        dtl::ConstScalarArg>;

class FusedAddRightScalarMultiply : public VectorKernelBase<
                                           FusedAddRightScalarMultiply,
                                           dtl::MutableVectorArg,
                                           dtl::ConstVectorArg,
                                           dtl::ConstScalarArg>
{
public:
    using VectorKernelBase::VectorKernelBase;

    string_view kernel_name() const noexcept;

    dtl::GenericBinaryFunction generic_op() const noexcept
    {
        return [](scalars::Scalar& out,
                  const scalars::Scalar& left,
                  const scalars::Scalar& multiplier) {
            out += (left * multiplier);
        };
    }
};

}// namespace algebra
}// namespace rpy

#endif// FUSED_ADD_RIGHT_SCALAR_MULTIPLY_H
