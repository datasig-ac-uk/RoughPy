#include "stream.h"



#include <roughpy/streams/stream.h>

using namespace rpy;
using namespace rpy::streams;

static const char* STREAM_DOC = R"rpydoc(
A stream is an abstract stream of data viewed as a rough path.
)rpydoc";


void python::init_stream(py::module_ &m) {

    py::class_<Stream> klass(m, "Stream", STREAM_DOC);

    // TODO: finish this off.

}
