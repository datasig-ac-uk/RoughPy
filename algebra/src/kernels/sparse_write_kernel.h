//
// Created by sam on 4/17/24.
//

#ifndef SPARSE_WRITE_KERNEL_H
#define SPARSE_WRITE_KERNEL_H

#include "argument_specs.h"
#include "generic_kernel.h"
#include "kernel.h"

namespace rpy {
namespace algebra {

class SparseWriteKernel;

extern template class VectorKernelBase<
        SparseWriteKernel,
        dtl::MutableVectorArg,
        dtl::ConstVectorArg>;

class SparseWriteKernel : public VectorKernelBase<SparseWriteKernel, dtl::MutableVectorArg, dtl::ConstVectorArg>
{
public:

    using VectorKernelBase::VectorKernelBase;

    string_view kernel_name() const noexcept;

    dtl::GenericUnaryFunction generic_op() const noexcept {
        return [](scalars::Scalar& out, const scalars::Scalar& in) {
            out = in;
        };
    }

};


}// namespace algebra
}// namespace rpy

#endif// SPARSE_WRITE_KERNEL_H
