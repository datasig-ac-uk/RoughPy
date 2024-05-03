
#ifndef ALGEBRA_HOST_KERNEL_UMINUS_H
#define ALGEBRA_HOST_KERNEL_UMINUS_H

#include "vector_unary_operator.h"
#include <roughpy/device_support/host_kernel.h>
#include <roughpy/device_support/operators.h>

#include <roughpy/core/macros.h>

namespace rpy { namespace algebra {

extern template class VectorUnaryOperator<rpy::devices::operators::Uminus, float>;
extern template class VectorUnaryOperator<rpy::devices::operators::Uminus, double>;

}

namespace devices {

extern template class HostKernel<algebra::VectorUnaryOperator<rpy::devices::operators::Uminus, float>>;
extern template class HostKernel<algebra::VectorUnaryOperator<rpy::devices::operators::Uminus, double>>;

}
}

#endif //ALGEBRA_HOST_KERNEL_UMINUS_H
