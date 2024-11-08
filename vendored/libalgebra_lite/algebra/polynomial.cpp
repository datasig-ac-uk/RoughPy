//
// Created by user on 30/08/22.
//

#include "libalgebra_lite/polynomial.h"

namespace lal {

template class polynomial<double_field>;
template class polynomial<float_field>;

#ifdef LAL_ENABLE_RATIONAL_COEFFS
template class polynomial<rational_field>;
#endif
}
