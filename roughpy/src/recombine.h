#ifndef RPY_PY__RECOMBINE_H_
#define RPY_PY__RECOMBINE_H_

#include <pybind11/pybind11.h>


namespace rpy {
namespace python {

namespace py = pybind11;

void init_recombine(py::module_& m);

} // namespace python
} // namespace rpy

#endif // RPY_PY__RECOMBINE_H_
