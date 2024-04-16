//
// Created by sam on 16/04/24.
//

#include "inplace_subtraction_kernel.h"
#include "generic_kernel_MV_CV.h"

namespace rpy {
namespace algebra {

template class VectorKernelBase<
        InplaceSubtractionKernel,
        dtl::MutableVectorArg,
        dtl::ConstVectorArg>;

}// namespace algebra
}// namespace rpy
