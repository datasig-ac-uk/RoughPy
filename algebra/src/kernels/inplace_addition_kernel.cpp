//
// Created by sam on 16/04/24.
//

#include "inplace_addition_kernel.h"

namespace rpy {
namespace algebra {

template class VectorKernelBase<
        InplaceAdditionKernel,
        dtl::MutableVectorArg,
        dtl::ConstVectorArg>;

}// namespace algebra
}// namespace rpy