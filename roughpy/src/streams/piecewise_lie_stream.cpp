//
// Created by user on 18/03/23.
//

#include "piecewise_lie_stream.h"

#include <roughpy/streams/stream.h>
#include <roughpy/streams/piecewise_lie_stream.h>

using namespace rpy;
using namespace pybind11::literals;

static const char* PW_LIE_STREAM_DOC = R"rpydoc(A stream formed of a sequence of interval-Lie pairs.
)rpydoc";


void python::init_piecewise_lie_stream(py::module_ &m) {

    py::class_<streams::PiecewiseLieStream> klass(m, "PiecewiseLieStream", PW_LIE_STREAM_DOC);

}
