//
// Created by sammorley on 25/11/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_RATIONAL_CONVERSION_H
#define ROUGHPY_GENERICS_INTERNAL_RATIONAL_CONVERSION_H


#include <memory>

#include "roughpy/core/macros.h"

#include "roughpy/generics/type_ptr.h"
#include "roughpy/generics/conversion_trait.h"

#include "generics/conversion_impl.h"

namespace rpy {
namespace generics::conv {


boost::container::flat_map<hash_t, std::unique_ptr<const ConversionFactory>>
make_mprational_conversion_to_table();

boost::container::flat_map<hash_t, std::unique_ptr<const ConversionFactory>>
make_mprational_conversion_from_table();


} // generics
} // rpy

#endif //ROUGHPY_GENERICS_INTERNAL_RATIONAL_CONVERSION_H
