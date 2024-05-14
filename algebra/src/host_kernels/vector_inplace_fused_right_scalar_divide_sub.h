
#ifndef ALGEBRA_HOST_KERNEL_FUSEDRIGHTSCALARDIVIDESUB_H
#define ALGEBRA_HOST_KERNEL_FUSEDRIGHTSCALARDIVIDESUB_H

#include "vector_binary_operator.h"
#include <roughpy/device_support/host_kernel.h>
#include <roughpy/device_support/operators.h>

#include <roughpy/core/macros.h>

namespace rpy { namespace algebra {



}

namespace devices {

extern template class HostKernel<algebra::VectorInplaceBinaryWithScalarOperator<rpy::devices::operators::FusedRightScalarDivideSub, float>>;
extern template class HostKernel<algebra::VectorInplaceBinaryWithScalarOperator<rpy::devices::operators::FusedRightScalarDivideSub, double>>;

}
}

#endif //ALGEBRA_HOST_KERNEL_FUSEDRIGHTSCALARDIVIDESUB_H
