//
// Created by sam on 14/04/23.
//

#include "externally_sourced_stream.h"

#include <roughpy/core/helpers.h>
#include <roughpy/core/macros.h>
#include <roughpy/platform.h>
#include <roughpy/streams/external_data_stream.h>
#include <roughpy/streams/stream.h>

#include "args/kwargs_to_path_metadata.h"
#include "scalars/scalar_type.h"
#include "scalars/scalars.h"
#include "stream.h"

#include <algorithm>

using namespace rpy;
using namespace rpy::streams;
using namespace pybind11::literals;

static const char* EXTERNALLY_SOURCED_STREAM_DOC
        = R"rpydoc(A stream that acquires its data dynamically from an external source.
)rpydoc";

static py::object
external_stream_constructor(string uri_string, const py::kwargs& kwargs)
{
    const auto pmd = python::kwargs_to_metadata(kwargs);

    url uri;
    auto uri_result = parse_uri_reference(uri_string);

    if (!uri_result) {

        try {
            auto path = fs::path(uri_string);
            uri_string = fs::absolute(path).string();
#ifdef RPY_PLATFORM_WINDOWS
            std::replace(uri_string.begin(), uri_string.end(), '\\', '/');
#endif
            if (fs::exists(path)) {
                uri = url();
                uri.set_scheme_id(URIScheme::file);
                uri.set_path(uri_string);
            }
        } catch (...) {
            RPY_THROW(py::value_error,
                    "could not parse path " + uri_string + " error code "
                    + uri_result.error().message()
            );
        }

    } else {
        uri = *uri_result;
    }

    auto factory = streams::ExternalDataStream::get_factory_for(uri);

    if (!factory) {
        RPY_THROW(py::value_error, "The uri " + uri_string + " is not supported");
    }

    //    factory.add_metadata({
    //        pmd.width,
    //        pmd.support,
    //        pmd.ctx,
    //        pmd.scalar_type,
    //        pmd.vector_type,
    //        pmd.resolution
    //    });

    if (pmd.width != 0) { factory.set_width(pmd.width); }
    if (pmd.depth != 0) { factory.set_depth(pmd.depth); }
    if (pmd.scalar_type != nullptr) { factory.set_ctype(pmd.scalar_type); }
    if (pmd.ctx) { factory.set_context(pmd.ctx); }
    if (pmd.resolution != 0) { factory.set_resolution(pmd.resolution); }
    if (pmd.support) { factory.set_support(*pmd.support); }
    if (pmd.vector_type) { factory.set_vtype(*pmd.vector_type); }
    if (pmd.schema) { factory.set_schema(pmd.schema); }

    PyObject* py_stream = python::RPyStream_FromStream(factory.construct());

    return py::reinterpret_steal<py::object>(py_stream);
}

void python::init_externally_sourced_stream(py::module_& m)
{

    py::class_<ExternalDataStream> klass(
            m, "ExternalDataStream", EXTERNALLY_SOURCED_STREAM_DOC
    );

    klass.def_static("from_uri", external_stream_constructor, "uri"_a);
}
