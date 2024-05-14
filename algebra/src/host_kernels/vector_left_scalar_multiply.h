
#ifndef ALGEBRA_HOST_KERNEL_LEFTSCALARMULTIPLY_H
#define ALGEBRA_HOST_KERNEL_LEFTSCALARMULTIPLY_H

#include "vector_unary_operator.h"
#include <roughpy/device_support/host_kernel.h>
#include <roughpy/device_support/operators.h>

#include <roughpy/core/macros.h>

namespace rpy { namespace algebra {



}

namespace devices {

extern template class HostKernel<algebra::VectorUnaryWithScalarOperator<rpy::devices::operators::LeftScalarMultiply, float>>;
extern template class HostKernel<algebra::VectorUnaryWithScalarOperator<rpy::devices::operators::LeftScalarMultiply, double>>;

}
}

#endif //ALGEBRA_HOST_KERNEL_LEFTSCALARMULTIPLY_H
