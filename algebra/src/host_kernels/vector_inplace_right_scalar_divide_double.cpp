
#include "vector_inplace_right_scalar_divide.h"

using namespace rpy;
using namespace rpy::algebra;

namespace rpy { namespace algebra {

//template class VectorInplaceUnaryWithScalarOperator<rpy::devices::operators::RightScalarDivide, double>;

}

namespace devices {

template class HostKernel<algebra::VectorInplaceUnaryWithScalarOperator<
    rpy::devices::operators::RightScalarDivide, double>>;

}
}
