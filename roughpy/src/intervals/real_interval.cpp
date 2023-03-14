#include "real_interval.h"

#include <sstream>

#include <roughpy/intervals/interval.h>
#include <roughpy/intervals/real_interval.h>

using namespace rpy;
using namespace rpy::intervals;
using namespace pybind11::literals;

static const char *REAL_INTERVAL_DOC = R"edoc(A half-open interval in the real line.
)edoc";

void python::init_real_interval(py::module_ &m) {

    py::class_<RealInterval, Interval> klass(m, "RealInterval", REAL_INTERVAL_DOC);
//    klass.def(py::init<>());
//    klass.def(py::init<IntervalType>());
    klass.def(py::init<double, double>(), "inf"_a, "sup"_a);
    klass.def(py::init<double, double, IntervalType>(), "inf"_a, "sup"_a, "interval_type"_a);
    klass.def(py::init<const Interval &>(), "arg"_a);
    klass.def("__repr__", [](const RealInterval &arg) {
      std::stringstream ss;
      ss << "RealInterval(inf="
         << std::to_string(arg.inf())
         << ", sup="
         << std::to_string(arg.sup())
         << ", type=";
      if (arg.type() == IntervalType::Clopen) {
          ss << "clopen";
      } else {
          ss << "opencl";
      }
      ss << ')';
      return ss.str();
    });
}
