//
// Created by sam on 14/04/23.
//


#include "externally_sourced_stream.h"

#include <roughpy/core/helpers.h>
#include <roughpy/streams/external_data_stream.h>
#include <roughpy/streams/stream.h>
#include <roughpy/platform.h>

#include "args/kwargs_to_path_metadata.h"
#include "scalars/scalars.h"
#include "scalars/scalar_type.h"
#include "stream.h"


using namespace rpy;
using namespace rpy::streams;
using namespace pybind11::literals;


static const char* EXTERNALLY_SOURCED_STREAM_DOC = R"rpydoc(A stream that acquires its data dynamically from an external source.
)rpydoc";


static py::object external_stream_constructor(std::string uri_string, const py::kwargs& kwargs) {
    const auto pmd = python::kwargs_to_metadata(kwargs);

    url uri(uri_string);

    const auto* factory = streams::ExternalDataStream::get_factory_for(uri);

    if (factory == nullptr) {
        throw py::value_error("The uri " + uri_string + " is not supported");
    }

    StreamMetadata md {
        pmd.width,
        pmd.support,
        pmd.ctx,
        pmd.scalar_type,
        pmd.vector_type,
        pmd.resolution
    };

    PyObject* py_stream = python::RPyStream_FromStream(factory->construct_stream(uri, std::move(md)));

    return py::reinterpret_steal<py::object>(py_stream);
}

void python::init_externally_sourced_stream(py::module_ &m) {

    py::class_<ExternalDataStream> klass(m, "ExternalDataStream", EXTERNALLY_SOURCED_STREAM_DOC);

    klass.def_static("from_uri", external_stream_constructor, "uri"_a);

}
