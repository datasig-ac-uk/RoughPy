#include "tick_stream.h"
#include "intervals/interval.h"
#include "roughpy/algebra/lie.h"

#include <roughpy/streams/tick_stream.h>
#include <roughpy/streams/stream.h>

#include "args/kwargs_to_path_metadata.h"
#include "scalars/scalar_type.h"
#include "scalars/scalars.h"
#include "streams/stream.h"

using namespace rpy;
using namespace pybind11::literals;

static const char* TICK_STREAM_DOC = R"rpydoc(Tick stream
)rpydoc";


static py::object construct(const py::object& data, const py::kwargs& kwargs) {

    auto pmd = python::kwargs_to_metadata(kwargs);

    std::vector<param_t> indices;

    python::PyToBufferOptions options;
    options.type = pmd.scalar_type;
    options.max_nested = 2;
    options.allow_scalar = false;

    auto buffer = python::py_to_buffer(data, options);

    scalars::ScalarStream stream(options.type);
    stream.reserve_size(options.shape.size());


    auto result = streams::Stream(
        streams::TickStream(
            std::move(stream),
            std::vector<const key_type*>(),
            std::move(indices),
            pmd.resolution,
            {
                pmd.width,
                pmd.support,
                pmd.ctx,
                pmd.scalar_type,
                pmd.vector_type,
                pmd.resolution
            }
            )
        );

    return py::reinterpret_steal<py::object>(python::RPyStream_FromStream(std::move(result)));
}


void python::init_tick_stream(py::module_ &m) {

    py::class_<streams::TickStream> klass(m, "TickStream", TICK_STREAM_DOC);

    klass.def_static("from", &construct, "data"_a);

}
