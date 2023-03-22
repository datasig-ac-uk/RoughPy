#include "streams.h"

#include "BaseStream.h"
#include "stream.h"

#include "function_stream.h"
#include "lie_increment_stream.h"
#include "piecewise_lie_stream.h"
#include "tick_stream.h"

using namespace rpy;
using namespace rpy::python;


void python::init_streams(py::module_ &m) {

    py::options options;
    options.disable_function_signatures();

    init_base_stream(m);
    init_stream(m);

    init_lie_increment_stream(m);
    init_piecewise_lie_stream(m);
    init_function_stream(m);
    init_tick_stream(m);
}
