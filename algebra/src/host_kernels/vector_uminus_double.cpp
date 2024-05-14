
#include "vector_uminus.h"

using namespace rpy;
using namespace rpy::algebra;

namespace rpy { namespace algebra {

//template class VectorUnaryOperator<rpy::devices::operators::Uminus, double>;

}

namespace devices {

template class HostKernel<algebra::VectorUnaryOperator<
    rpy::devices::operators::Uminus, double>>;

}
}
