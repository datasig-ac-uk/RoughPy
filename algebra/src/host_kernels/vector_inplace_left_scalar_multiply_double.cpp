
#include "vector_inplace_left_scalar_multiply.h"

using namespace rpy;
using namespace rpy::algebra;

namespace rpy { namespace algebra {

//template class VectorInplaceUnaryWithScalarOperator<rpy::devices::operators::LeftScalarMultiply, double>;

}

namespace devices {

template class HostKernel<algebra::VectorInplaceUnaryWithScalarOperator<
    rpy::devices::operators::LeftScalarMultiply, double>>;

}
}
