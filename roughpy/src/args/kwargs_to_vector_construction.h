#ifndef RPY_PY_ARGS_KWARGS_TO_VECTOR_CONSTRUCTION_H_
#define RPY_PY_ARGS_KWARGS_TO_VECTOR_CONSTRUCTION_H_

#include <pybind11/pybind11.h>

#include <roughpy/algebra/algebra_fwd.h>
#include <roughpy/algebra/context_fwd.h>
#include <roughpy/scalars/scalar.h>

namespace rpy {
namespace python {

namespace py = pybind11;

struct PyVectorConstructionHelper {
    /// Context if provided by user
    algebra::context_pointer ctx;
    /// Width and depth
    deg_t width = 0;
    deg_t depth = 0;
    /// Coefficient type
    const scalars::ScalarType *ctype = nullptr;
    /// Vector type to be requested
    algebra::VectorType vtype = algebra::VectorType::Dense;
    /// flags for saying if the user explicitly requested ctype and vtype
    bool ctype_requested = false;
    bool vtype_requested = false;
    /// Data type provided
};

PyVectorConstructionHelper kwargs_to_construction_data(const py::kwargs &kwargs);

}// namespace python
}// namespace rpy

#endif// RPY_PY_ARGS_KWARGS_TO_VECTOR_CONSTRUCTION_H_
