
#include "vector_inplace_sub.h"

using namespace rpy;
using namespace rpy::algebra;

namespace rpy { namespace algebra {

//template class VectorInplaceBinaryOperator<rpy::devices::operators::Sub, double>;

}

namespace devices {

template class HostKernel<algebra::VectorInplaceBinaryOperator<
    rpy::devices::operators::Sub, double>>;

}
}
