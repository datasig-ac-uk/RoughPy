#ifndef RPY_PY_ARGS_NUMPY_H_
#define RPY_PY_ARGS_NUMPY_H_
#ifdef ROUGHPY_WITH_NUMPY

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <roughpy/scalars/scalar.h>

#include <string>

namespace rpy {
namespace python {

namespace py = pybind11;

const scalars::ScalarType* npy_dtype_to_ctype(py::dtype dtype);

py::dtype ctype_to_npy_dtype(const scalars::ScalarType* type);


std::string npy_dtype_to_identifier(py::dtype dtype);

} // namespace python
} // namespace rpy

#endif // ROUGHPY_WITH_NUMPY
#endif // RPY_PY_ARGS_NUMPY_H_
