
#ifndef ALGEBRA_HOST_KERNEL_FUSEDLEFTSCALARMULTIPLYADD_H
#define ALGEBRA_HOST_KERNEL_FUSEDLEFTSCALARMULTIPLYADD_H

#include "vector_binary_operator.h"
#include <roughpy/device_support/host_kernel.h>
#include <roughpy/device_support/operators.h>

#include <roughpy/core/macros.h>

namespace rpy { namespace algebra {



}

namespace devices {

extern template class HostKernel<algebra::VectorInplaceBinaryWithScalarOperator<rpy::devices::operators::FusedLeftScalarMultiplyAdd, float>>;
extern template class HostKernel<algebra::VectorInplaceBinaryWithScalarOperator<rpy::devices::operators::FusedLeftScalarMultiplyAdd, double>>;

}
}

#endif //ALGEBRA_HOST_KERNEL_FUSEDLEFTSCALARMULTIPLYADD_H
