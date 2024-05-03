
#ifndef ALGEBRA_HOST_KERNEL_FUSEDRIGHTSCALARDIVIDEADD_H
#define ALGEBRA_HOST_KERNEL_FUSEDRIGHTSCALARDIVIDEADD_H

#include "vector_binary_operator.h"
#include <roughpy/device_support/host_kernel.h>
#include <roughpy/device_support/operators.h>

#include <roughpy/core/macros.h>

namespace rpy { namespace algebra {

extern template class VectorInplaceBinaryWithScalarOperator<rpy::devices::operators::FusedRightScalarDivideAdd, float>;
extern template class VectorInplaceBinaryWithScalarOperator<rpy::devices::operators::FusedRightScalarDivideAdd, double>;

}

namespace devices {

extern template class HostKernel<algebra::VectorInplaceBinaryWithScalarOperator<rpy::devices::operators::FusedRightScalarDivideAdd, float>>;
extern template class HostKernel<algebra::VectorInplaceBinaryWithScalarOperator<rpy::devices::operators::FusedRightScalarDivideAdd, double>>;

}
}

#endif //ALGEBRA_HOST_KERNEL_FUSEDRIGHTSCALARDIVIDEADD_H
