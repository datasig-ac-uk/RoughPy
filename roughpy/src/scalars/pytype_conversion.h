//
// Created by sam on 2/29/24.
//

#ifndef ROUGHPY_PYTYPE_CONVERSION_H
#define ROUGHPY_PYTYPE_CONVERSION_H

#include "roughpy_module.h"

#include <roughpy/scalars/scalars_fwd.h>

namespace rpy {
namespace python {

RPY_NO_DISCARD bool
is_imported_type(py::handle obj, const char* module, const char* type_name);

RPY_NO_DISCARD bool is_py_scalar_type(py::handle obj);

RPY_NO_DISCARD const scalars::ScalarType* type_of_pyscalar(py::handle obj);

RPY_NO_DISCARD const scalars::ScalarType* py_type_to_type(py::handle obj);

RPY_NO_DISCARD
devices::TypeInfo py_type_to_type_info(py::handle pytype);

py::object init_scalar_mapping();

}// namespace python
}// namespace rpy

#endif// ROUGHPY_PYTYPE_CONVERSION_H
