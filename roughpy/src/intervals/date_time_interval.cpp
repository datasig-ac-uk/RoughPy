//
// Created by user on 04/07/23.
//

#include "date_time_interval.h"

#include <datetime.h>

using namespace rpy;
using namespace intervals;

python::DateTimeInterval::DateTimeInterval(py::object dt_begin,
                                           py::object dt_end)
    : Interval(IntervalType::Clopen), m_dt_begin(std::move(dt_begin)),
      m_dt_end(std::move(dt_end))
{
    if (Py_TYPE(m_dt_begin.ptr()) != Py_TYPE(m_dt_end.ptr())) {
        RPY_THROW(py::type_error,"both begin and end objects must have the same "
                             "type");
    }

    if (!is_py_datetime(m_dt_begin) || !is_py_date(m_dt_begin)
        || !is_py_time(m_dt_begin)) {
        RPY_THROW(py::type_error,"begin and end must be datetime, data, or time "
                             "objects");
    }
}

param_t python::DateTimeInterval::inf() const { return 0.0; }
param_t python::DateTimeInterval::sup() const
{
    python::PyDateTimeConversionOptions options{PyDateTimeResolution::Seconds};

    return python::convert_delta_from_datetimes(m_dt_end, m_dt_begin, options);
}



void python::init_datetime_interval(py::module_& m)
{

    py::class_<DateTimeInterval, intervals::Interval> cls(m,
                                                          "DateTimeInterval");

    cls.def("dt_inf", &DateTimeInterval::dt_inf);
    cls.def("dt_sup", &DateTimeInterval::dt_sup);
}
