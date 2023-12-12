//
// Created by sam on 12/11/23.
//

#include "rational_coefficients.h"


namespace lal {

template class coefficient_field<rpy::devices::rational_scalar_type>;

template class coefficient_ring<rpy::devices::rational_poly_scalar, rpy::devices::rational_scalar_type>;

}
