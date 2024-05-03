
#ifndef ALGEBRA_HOST_KERNEL_FUSEDRIGHTSCALARMULTIPLYSUB_H
#define ALGEBRA_HOST_KERNEL_FUSEDRIGHTSCALARMULTIPLYSUB_H

#include "vector_binary_operator.h"
#include <roughpy/device_support/host_kernel.h>
#include <roughpy/device_support/operators.h>

#include <roughpy/core/macros.h>

namespace rpy { namespace algebra {

extern template class VectorInplaceBinaryWithScalarOperator<rpy::devices::operators::FusedRightScalarMultiplySub, float>;
extern template class VectorInplaceBinaryWithScalarOperator<rpy::devices::operators::FusedRightScalarMultiplySub, double>;

}

namespace devices {

extern template class HostKernel<algebra::VectorInplaceBinaryWithScalarOperator<rpy::devices::operators::FusedRightScalarMultiplySub, float>>;
extern template class HostKernel<algebra::VectorInplaceBinaryWithScalarOperator<rpy::devices::operators::FusedRightScalarMultiplySub, double>>;

}
}

#endif //ALGEBRA_HOST_KERNEL_FUSEDRIGHTSCALARMULTIPLYSUB_H
