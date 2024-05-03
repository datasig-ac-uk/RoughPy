
#ifndef ALGEBRA_HOST_KERNEL_FUSEDRIGHTSCALARMULTIPLYADD_H
#define ALGEBRA_HOST_KERNEL_FUSEDRIGHTSCALARMULTIPLYADD_H

#include "vector_binary_operator.h"
#include <roughpy/device_support/host_kernel.h>
#include <roughpy/device_support/operators.h>

#include <roughpy/core/macros.h>

namespace rpy { namespace algebra {

extern template class VectorInplaceBinaryWithScalarOperator<rpy::devices::operators::FusedRightScalarMultiplyAdd, float>;
extern template class VectorInplaceBinaryWithScalarOperator<rpy::devices::operators::FusedRightScalarMultiplyAdd, double>;

}

namespace devices {

extern template class HostKernel<algebra::VectorInplaceBinaryWithScalarOperator<rpy::devices::operators::FusedRightScalarMultiplyAdd, float>>;
extern template class HostKernel<algebra::VectorInplaceBinaryWithScalarOperator<rpy::devices::operators::FusedRightScalarMultiplyAdd, double>>;

}
}

#endif //ALGEBRA_HOST_KERNEL_FUSEDRIGHTSCALARMULTIPLYADD_H
