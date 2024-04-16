//
// Created by sam on 16/04/24.
//

#include "inplace_subtraction_kernel.h"


namespace rpy {
namespace algebra {

template class VectorKernelBase<InplaceSubtractionKernel, dtl::MutableVectorArg, dtl::ConstVectorArg>;


} // algebra
} // rpy