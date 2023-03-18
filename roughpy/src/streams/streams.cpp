#include "streams.h"

#include "BaseStream.h"
#include "stream.h"



using namespace rpy;
using namespace rpy::python;


void python::init_streams(py::module_ &m) {

    py::options options;
    options.disable_function_signatures();

    init_base_stream(m);
    init_stream(m);


}
