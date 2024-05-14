
#include "vector_sub.h"

using namespace rpy;
using namespace rpy::algebra;

namespace rpy { namespace algebra {

//template class VectorBinaryOperator<rpy::devices::operators::Sub, float>;

}

namespace devices {

template class HostKernel<algebra::VectorBinaryOperator<
    rpy::devices::operators::Sub, float>>;

}
}
