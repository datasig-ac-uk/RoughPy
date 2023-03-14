//
// Created by user on 14/03/23.
//

#ifndef ROUGHPY_ROUGHPY_SRC_ALGEBRA_ALGEBRA_H
#define ROUGHPY_ROUGHPY_SRC_ALGEBRA_ALGEBRA_H

#include <pybind11/pybind11.h>


namespace rpy { namespace python {

namespace py = pybind11;

void init_algebra(py::module_& m);


}}

#endif//ROUGHPY_ROUGHPY_SRC_ALGEBRA_ALGEBRA_H
