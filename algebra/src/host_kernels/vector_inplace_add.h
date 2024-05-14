
#ifndef ALGEBRA_HOST_KERNEL_ADD_H
#define ALGEBRA_HOST_KERNEL_ADD_H

#include "vector_binary_operator.h"
#include <roughpy/device_support/host_kernel.h>
#include <roughpy/device_support/operators.h>

#include <roughpy/core/macros.h>

namespace rpy { namespace algebra {



}

namespace devices {

extern template class HostKernel<algebra::VectorInplaceBinaryOperator<rpy::devices::operators::Add, float>>;
extern template class HostKernel<algebra::VectorInplaceBinaryOperator<rpy::devices::operators::Add, double>>;

}
}

#endif //ALGEBRA_HOST_KERNEL_ADD_H
