//
// Created by sam on 12/11/23.
//

#ifndef ROUGHPY_RATIONAL_COEFFICIENTS_H
#define ROUGHPY_RATIONAL_COEFFICIENTS_H

#include <roughpy/platform/devices/rational_numbers.h>

#include <libalgebra_lite/coefficients.h>

namespace lal {

extern template struct coefficient_field<rpy::devices::rational_poly_scalar>;

extern template struct coefficient_ring<rpy::devices::rational_poly_scalar,
rpy::devices::rational_scalar_type>;
}

namespace rpy {
namespace algebra {


using rational_field = lal::coefficient_field<devices::rational_scalar_type>;

using rational_poly_ring = lal::coefficient_ring<devices::rational_poly_scalar, devices::rational_scalar_type>;

}
}


#endif //ROUGHPY_RATIONAL_COEFFICIENTS_H
