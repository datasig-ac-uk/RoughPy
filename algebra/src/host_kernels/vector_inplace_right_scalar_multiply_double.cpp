
#include "vector_inplace_right_scalar_multiply.h"

using namespace rpy;
using namespace rpy::algebra;

namespace rpy { namespace algebra {

template class VectorInplaceUnaryWithScalarOperator<rpy::devices::operators::RightScalarMultiply, double>;

}

namespace devices {

template class HostKernel<algebra::VectorInplaceUnaryWithScalarOperator<
    rpy::devices::operators::RightScalarMultiply, double>>;

}
}
