#ifndef RPY_PY_ARGS_KWARGS_TO_PATH_METADATA_H_
#define RPY_PY_ARGS_KWARGS_TO_PATH_METADATA_H_

#include <pybind11/pybind11.h>

#include <roughpy/streams/stream_base.h>

namespace rpy {
namespace python {

namespace py = pybind11;

struct PyStreamMetaData {
    deg_t width;
    deg_t depth;
    intervals::RealInterval support;
    algebra::context_pointer ctx;
    const scalars::ScalarType* scalar_type;
    algebra::VectorType vector_type;
    streams::resolution_t resolution;
};


PyStreamMetaData kwargs_to_metadata(const py::kwargs& kwargs);


} // namespace python
} // namespace rpy

#endif // RPY_PY_ARGS_KWARGS_TO_PATH_METADATA_H_
