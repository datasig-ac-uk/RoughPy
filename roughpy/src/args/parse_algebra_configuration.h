//
// Created by sam on 2/19/24.
//

#ifndef ROUGHPY_PARSE_ALGEBRA_CONFIGURATION_H
#define ROUGHPY_PARSE_ALGEBRA_CONFIGURATION_H


#include "roughpy_module.h"
#include "algebra/context.h"
#include "scalars/scalar_type.h"

#include <roughpy/core/types.h>
#include <roughpy/algebra/context_fwd.h>

namespace rpy {
namespace python {

struct AlgebraConfiguration {
    algebra::context_pointer ctx;
    optional <deg_t> width;
    optional <deg_t> depth;
    PyScalarMetaType* scalar_type;
};

AlgebraConfiguration parse_algebra_configuration(py::kwargs& kwargs);

}
}

#endif //ROUGHPY_PARSE_ALGEBRA_CONFIGURATION_H
