//
// Created by user on 20/06/23.
//

#include "convert_timestamp.h"

rpy::param_t rpy::python::convert_timestamp(const py::object &py_timestamp) {
    // TODO: Implement this properly to handle python datetime objects
    return py_timestamp.cast<rpy::param_t>();
}
