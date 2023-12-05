//
// Created by sam on 11/16/23.
//

#ifndef ROUGHPY_SCALARS_PRINT_H
#define ROUGHPY_SCALARS_PRINT_H

#include "scalar.h"
#include "do_macro.h"
#include <ostream>

namespace rpy {
namespace scalars {
namespace dtl {


void print_scalar_val(std::ostream& os,
                      const void* ptr,
                      const devices::TypeInfo& info);


}
}
}


#endif // ROUGHPY_SCALARS_PRINT_H
