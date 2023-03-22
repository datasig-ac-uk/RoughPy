#ifndef RPY_PY_STREAMS_FUNCTION_PATH_H_
#define RPY_PY_STREAMS_FUNCTION_PATH_H_

#include "roughpy_module.h"

#include <roughpy/streams/dynamically_constructed_stream.h>

namespace rpy {
namespace python {

class FunctionStream : public streams::DynamicallyConstructedStream {
    py::function m_fn;
public:
    FunctionStream(py::function fn, streams::StreamMetadata md);

protected:
    algebra::Lie eval(const intervals::Interval &interval) const override;
};


void init_function_stream(py::module_& m);


} // namespace python
} // namespace rpy

#endif // RPY_PY_STREAMS_FUNCTION_PATH_H_
