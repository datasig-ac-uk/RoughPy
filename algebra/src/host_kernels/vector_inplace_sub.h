
#ifndef ALGEBRA_HOST_KERNEL_SUB_H
#define ALGEBRA_HOST_KERNEL_SUB_H

#include "vector_binary_operator.h"
#include <roughpy/device_support/host_kernel.h>
#include <roughpy/device_support/operators.h>

#include <roughpy/core/macros.h>

namespace rpy { namespace algebra {

extern template class VectorInplaceBinaryOperator<rpy::devices::operators::Sub, float>;
extern template class VectorInplaceBinaryOperator<rpy::devices::operators::Sub, double>;

}

namespace devices {

extern template class HostKernel<algebra::VectorInplaceBinaryOperator<rpy::devices::operators::Sub, float>>;
extern template class HostKernel<algebra::VectorInplaceBinaryOperator<rpy::devices::operators::Sub, double>>;

}
}

#endif //ALGEBRA_HOST_KERNEL_SUB_H
