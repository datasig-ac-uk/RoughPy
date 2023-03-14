#ifndef RPY_PY_STREAMS_STREAMS_H_
#define RPY_PY_STREAMS_STREAMS_H_

#include <pybind11/pybind11.h>

namespace rpy {
namespace python {

namespace py = pybind11;

void init_streams(py::module_& m);

} // namespace python
} // namespace rpy

#endif // RPY_PY_STREAMS_STREAMS_H_
