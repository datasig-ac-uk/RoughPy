//
// Created by user on 11/03/23.
//

#include <pybind11/pybind11.h>


#include "scalars/scalars.h"
#include "intervals/intervals.h"
#include "algebra/algebra.h"
#include "streams/streams.h"
#include "recombine.h"


#ifndef ROUGHPY_VERSION_STRING
#define ROUGHPY_VERSION_STRING "1.0.0"
#endif

namespace py = pybind11;

PYBIND11_MODULE(_roughpy, m) {
    using namespace rpy::python;

    m.add_object("__version__", py::str(ROUGHPY_VERSION_STRING));

    init_scalars(m);
    init_intervals(m);
    init_algebra(m);
    init_streams(m);
    init_recombine(m);

}
