// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//
// Created by sam on 14/04/23.
//

#include "externally_sourced_stream.h"

#include "roughpy/core/check.h"                    // for throw_exception
#include "roughpy/core/types.h"                    // for string
#include <roughpy/core/helpers.h>
#include <roughpy/core/macros.h>

#include <roughpy/platform.h>
#include <roughpy/streams/external_data_stream.h>
#include <roughpy/streams/stream.h>

#include "args/kwargs_to_path_metadata.h"
#include "scalars/scalar_type.h"
#include "scalars/scalars.h"
#include "stream.h"
#include "streams/schema_finalization.h"

#include <boost/url/parse.hpp>

#include <algorithm>

using namespace rpy;
using namespace rpy::streams;
using namespace pybind11::literals;

using boost::urls::url;
using boost::urls::parse_uri_reference;
using URIScheme = boost::urls::scheme;

static const char* EXTERNALLY_SOURCED_STREAM_DOC
        = R"rpydoc(A :class:`~Stream` that acquires its :py:attr:`~data` dynamically from an external source.
)rpydoc";

static py::object
external_stream_constructor(string uri_string, py::kwargs kwargs)
{
    auto pmd = python::kwargs_to_metadata(kwargs);

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
    if (!pmd.resolution) {
        pmd.resolution = 0;
    }

    if (pmd.width != 0) { factory.set_width(pmd.width); }
    if (pmd.depth != 0) { factory.set_depth(pmd.depth); }
    if (pmd.scalar_type != nullptr) { factory.set_ctype(pmd.scalar_type); }
    if (pmd.ctx) { factory.set_context(pmd.ctx); }
    if (pmd.resolution != 0) { factory.set_resolution(*pmd.resolution); }
    if (pmd.support) { factory.set_support(*pmd.support); }
    if (pmd.vector_type) { factory.set_vtype(*pmd.vector_type); }

    factory.set_schema(pmd.schema);

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
