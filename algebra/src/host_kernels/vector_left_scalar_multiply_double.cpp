
#include "vector_left_scalar_multiply.h"

using namespace rpy;
using namespace rpy::algebra;

namespace rpy { namespace algebra {

//template class VectorUnaryWithScalarOperator<rpy::devices::operators::LeftScalarMultiply, double>;

}

namespace devices {

template class HostKernel<algebra::VectorUnaryWithScalarOperator<
    rpy::devices::operators::LeftScalarMultiply, double>>;

}
}
