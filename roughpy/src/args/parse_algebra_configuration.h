//
// Created by sam on 2/19/24.
//

#ifndef ROUGHPY_PARSE_ALGEBRA_CONFIGURATION_H
#define ROUGHPY_PARSE_ALGEBRA_CONFIGURATION_H


#include "roughpy_module.h"

#include <roughpy/core/types.h>
#include <roughpy/scalars/scalars_fwd.h>
#include <roughpy/algebra/context_fwd.h>

namespace rpy {
namespace python {

struct AlgebraConfiguration {
    algebra::context_pointer ctx;
    optional <deg_t> width;
    optional <deg_t> depth;
    const scalars::ScalarType* scalar_type;
};

RPY_NO_EXPORT
AlgebraConfiguration parse_algebra_configuration(py::kwargs& kwargs);

}
}

#endif //ROUGHPY_PARSE_ALGEBRA_CONFIGURATION_H
