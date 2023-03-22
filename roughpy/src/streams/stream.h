#ifndef RPY_PY_STREAMS_STREAM_H_
#define RPY_PY_STREAMS_STREAM_H_

#include "roughpy_module.h"

#include <roughpy/streams/stream.h>

namespace rpy {
namespace python {

extern "C" {
struct RPyStream {
    PyObject_VAR_HEAD;
    streams::Stream m_data;
};

extern PyTypeObject RPyStream_Type;
}

PyObject* RPyStream_FromStream(streams::Stream&& stream);

void init_stream(py::module_ &m);

}// namespace python
}// namespace rpy

#endif// RPY_PY_STREAMS_STREAM_H_
