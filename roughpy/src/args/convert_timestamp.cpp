//
// Created by user on 20/06/23.
//

#include "convert_timestamp.h"

// Include Python's datetime module
#include <datetime.h>

#include "numpy.h"

namespace {

using options_t = rpy::python::PyDateTimeConversionOptions;

inline py::object get_py_reference_delta(rpy::python::PyDateTimeResolution resolution) {
    switch (resolution) {
        case rpy::python::PyDateTimeResolution::Microseconds:
            return py::reinterpret_steal<py::object>(PyDelta_FromDSU(0, 0, 1));
        case rpy::python::PyDateTimeResolution::Milliseconds:
            return py::reinterpret_steal<py::object>(PyDelta_FromDSU(0, 0, 1000));
        case rpy::python::PyDateTimeResolution::Seconds:
            return py::reinterpret_steal<py::object>(PyDelta_FromDSU(0, 1, 0));
        case rpy::python::PyDateTimeResolution::Minutes:
            return py::reinterpret_steal<py::object>(PyDelta_FromDSU(0, 60, 0));
        case rpy::python::PyDateTimeResolution::Hours:
            return py::reinterpret_steal<py::object>(PyDelta_FromDSU(0, 3600, 0));
        case rpy::python::PyDateTimeResolution::Days:
            return py::reinterpret_steal<py::object>(PyDelta_FromDSU(1, 0, 0));
    }
}

rpy::param_t pytimedelta_to_param(PyObject* py_time_delta, const options_t& options) {

    if (PyDelta_Check(py_time_delta)) {
        // Python datetime object
        auto reference = get_py_reference_delta(options.resolution);
        auto* py_reference = reference.ptr();

        auto normalised_delta = py::reinterpret_steal<py::object>(PyNumber_TrueDivide(py_time_delta, py_reference));


    }

    return 0;
}


}

rpy::param_t rpy::python::convert_delta_from_datetimes(py::handle current, py::handle previous, const rpy::python::PyDateTimeConversionOptions &options) {
    return 0;
}
rpy::param_t rpy::python::convert_timedelta(py::handle py_timedelta, const rpy::python::PyDateTimeConversionOptions &options) {
    return 0;
}

void rpy::python::init_datetime(py::module_ &m) {
    PyDateTime_IMPORT;
}
