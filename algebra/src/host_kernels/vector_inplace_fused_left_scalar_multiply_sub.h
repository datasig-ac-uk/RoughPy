
#ifndef ALGEBRA_HOST_KERNEL_FUSEDLEFTSCALARMULTIPLYSUB_H
#define ALGEBRA_HOST_KERNEL_FUSEDLEFTSCALARMULTIPLYSUB_H

#include "vector_binary_operator.h"
#include <roughpy/device_support/host_kernel.h>
#include <roughpy/device_support/operators.h>

#include <roughpy/core/macros.h>

namespace rpy { namespace algebra {



}

namespace devices {

extern template class HostKernel<algebra::VectorInplaceBinaryWithScalarOperator<rpy::devices::operators::FusedLeftScalarMultiplySub, float>>;
extern template class HostKernel<algebra::VectorInplaceBinaryWithScalarOperator<rpy::devices::operators::FusedLeftScalarMultiplySub, double>>;

}
}

#endif //ALGEBRA_HOST_KERNEL_FUSEDLEFTSCALARMULTIPLYSUB_H
