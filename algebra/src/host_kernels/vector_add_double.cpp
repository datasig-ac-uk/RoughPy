
#include "vector_add.h"

using namespace rpy;
using namespace rpy::algebra;

namespace rpy { namespace algebra {

template class VectorBinaryOperator<rpy::devices::operators::Add, double>;

}

namespace devices {

template class HostKernel<algebra::VectorBinaryOperator<
    rpy::devices::operators::Add, double>>;

}
}
