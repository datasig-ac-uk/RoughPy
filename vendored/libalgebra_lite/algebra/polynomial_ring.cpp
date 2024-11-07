//
// Created by user on 01/09/22.
//


#include "libalgebra_lite/coefficients.h"
#include "libalgebra_lite/polynomial.h"


namespace lal {

template struct coefficient_ring<polynomial<float_field>, float>;
template struct coefficient_ring<polynomial<double_field>, double>;

#ifdef LAL_ENABLE_RATIONAL_COEFFS
template struct coefficient_ring<polynomial<rational_field>, typename rational_field::scalar_type>;
#endif

}
