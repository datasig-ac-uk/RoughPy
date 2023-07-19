//
// Created by user on 20/06/23.
//

#include "convert_timestamp.h"

// Include Python's datetime module
#include <datetime.h>

#include "numpy.h"

bool rpy::python::is_py_datetime(py::handle object) noexcept
{
    return static_cast<bool>(PyDateTime_Check(object.ptr()));
}
bool rpy::python::is_py_date(py::handle object) noexcept
{
    return static_cast<bool>(PyDate_Check(object.ptr()));
}
bool rpy::python::is_py_time(py::handle object) noexcept
{
    return static_cast<bool>(PyTime_Check(object.ptr()));
}

namespace {

using options_t = rpy::python::PyDateTimeConversionOptions;

inline py::object
get_py_reference_delta(rpy::python::PyDateTimeResolution resolution)
{
    switch (resolution) {
        case rpy::python::PyDateTimeResolution::Microseconds:
            return py::reinterpret_steal<py::object>(PyDelta_FromDSU(0, 0, 1));
        case rpy::python::PyDateTimeResolution::Milliseconds:
            return py::reinterpret_steal<py::object>(PyDelta_FromDSU(0, 0, 1000)
            );
        case rpy::python::PyDateTimeResolution::Seconds:
            return py::reinterpret_steal<py::object>(PyDelta_FromDSU(0, 1, 0));
        case rpy::python::PyDateTimeResolution::Minutes:
            return py::reinterpret_steal<py::object>(PyDelta_FromDSU(0, 60, 0));
        case rpy::python::PyDateTimeResolution::Hours:
            return py::reinterpret_steal<py::object>(PyDelta_FromDSU(0, 3600, 0)
            );
        case rpy::python::PyDateTimeResolution::Days:
            return py::reinterpret_steal<py::object>(PyDelta_FromDSU(1, 0, 0));
    }
    RPY_UNREACHABLE_RETURN(py::object());
}

rpy::param_t
pytimedelta_to_param(PyObject* py_time_delta, const options_t& options)
{
    rpy::param_t result;
    if (PyDelta_Check(py_time_delta)) {
        // Python datetime object
        auto reference = get_py_reference_delta(options.resolution);
        auto* py_reference = reference.ptr();
        auto* normalised_delta
                = PyNumber_TrueDivide(py_time_delta, py_reference);
        if (normalised_delta == nullptr) { throw py::error_already_set(); }
        result = PyFloat_AsDouble(normalised_delta);
        Py_XDECREF(normalised_delta);
    } else if (PyLong_Check(py_time_delta)) {
        result = PyLong_AsDouble(py_time_delta);
    } else if (PyFloat_Check(py_time_delta)) {
        result = PyFloat_AsDouble(py_time_delta);
    } else {
        RPY_THROW(py::type_error, "expected datetime, int, or float");
    }

    return result;
}

py::object generic_to_timestamp(py::handle generic_timestamp)
{
    // We can just pass back generic timestamps that are actually timestamps
    if (PyDateTime_Check(generic_timestamp.ptr())
        || PyTime_Check(generic_timestamp.ptr())) {
        return py::reinterpret_borrow<py::object>(generic_timestamp);
    }

    // Integers should be considered as seconds since POSIX epoch
    if (PyLong_Check(generic_timestamp.ptr())) {
        auto* pdt = PyDateTime_FromTimestamp(generic_timestamp.ptr());
        if (pdt == nullptr) { throw py::error_already_set(); }
        return py::reinterpret_steal<py::object>(pdt);
    }

    // str objects should be parsed as if an ISO-8601 datetime string
    if (PyUnicode_Check(generic_timestamp.ptr())) {
        RPY_THROW(py::type_error,
                "currently conversion from ISO-8601 strings is not supported"
        );
    }

    RPY_THROW(py::type_error, "unsupported type for conversion to datetime");
}

}// namespace

rpy::param_t rpy::python::convert_delta_from_datetimes(
        py::handle current, py::handle previous,
        const rpy::python::PyDateTimeConversionOptions& options
)
{

    if (py::isinstance<py::float_>(current)
        && py::isinstance<py::float_>(previous)) {
        return current.cast<param_t>() - previous.cast<param_t>();
    }

    const py::object dt_current = generic_to_timestamp(current);
    const py::object dt_previous = generic_to_timestamp(previous);

    py::object delta = dt_current - dt_previous;
    return pytimedelta_to_param(delta.ptr(), options);
}
rpy::param_t rpy::python::convert_timedelta(
        py::handle py_timedelta,
        const rpy::python::PyDateTimeConversionOptions& options
)
{
    return 0;
}

void rpy::python::init_datetime(py::module_& m) { PyDateTime_IMPORT; }
