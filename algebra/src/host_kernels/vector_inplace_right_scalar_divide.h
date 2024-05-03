
#ifndef ALGEBRA_HOST_KERNEL_RIGHTSCALARDIVIDE_H
#define ALGEBRA_HOST_KERNEL_RIGHTSCALARDIVIDE_H

#include "vector_unary_operator.h"
#include <roughpy/device_support/host_kernel.h>
#include <roughpy/device_support/operators.h>

#include <roughpy/core/macros.h>

namespace rpy { namespace algebra {

extern template class VectorInplaceUnaryWithScalarOperator<rpy::devices::operators::RightScalarDivide, float>;
extern template class VectorInplaceUnaryWithScalarOperator<rpy::devices::operators::RightScalarDivide, double>;

}

namespace devices {

extern template class HostKernel<algebra::VectorInplaceUnaryWithScalarOperator<rpy::devices::operators::RightScalarDivide, float>>;
extern template class HostKernel<algebra::VectorInplaceUnaryWithScalarOperator<rpy::devices::operators::RightScalarDivide, double>>;

}
}

#endif //ALGEBRA_HOST_KERNEL_RIGHTSCALARDIVIDE_H
