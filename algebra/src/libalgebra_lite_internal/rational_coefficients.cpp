//
// Created by sam on 12/11/23.
//

#include "rational_coefficients.h"


namespace lal {

template struct coefficient_field<rpy::devices::rational_scalar_type>;

template struct coefficient_ring<rpy::devices::rational_poly_scalar, rpy::devices::rational_scalar_type>;

}
