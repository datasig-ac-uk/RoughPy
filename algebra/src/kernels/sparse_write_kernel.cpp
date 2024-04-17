//
// Created by sam on 4/17/24.
//

#include "sparse_write_kernel.h"
#include "generic_kernel_MV_CV.h"

namespace rpy {
namespace algebra {

template class VectorKernelBase<
        SparseWriteKernel,
        dtl::MutableVectorArg,
        dtl::ConstVectorArg>;

}// namespace algebra
}// namespace rpy

std::string_view rpy::algebra::SparseWriteKernel::kernel_name() const noexcept
{
    return "sparse_write";
}
