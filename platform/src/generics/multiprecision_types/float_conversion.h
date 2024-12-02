//
// Created by sam on 28/11/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_FLOAT_CONVERSION_H
#define ROUGHPY_GENERICS_INTERNAL_FLOAT_CONVERSION_H

#include <memory>

#include <boost/container/flat_map.hpp>

#include "roughpy/core/hash.h"

#include "roughpy/generics/conversion_trait.h"
#include "generics/conversion_impl.h"

namespace rpy::generics::conv {

boost::container::flat_map<hash_t, std::unique_ptr<const ConversionFactory>>
make_mpfloat_conversion_to_table();

boost::container::flat_map<hash_t, std::unique_ptr<const ConversionFactory>>
make_mpfloat_conversion_from_table();

}

#endif //ROUGHPY_GENERICS_INTERNAL_FLOAT_CONVERSION_H
